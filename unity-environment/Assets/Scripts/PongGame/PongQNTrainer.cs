using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class PongQNTrainer : InternalTrainer {

    public bool training = false;

    [Header("Training Settings")]
    public int experienceBufferSize = 5000000;
    public int batchSize = 64;
    public float discountFactor = 0.98f;
    public int trainingStepInterval = 10;
    public int stepsBeforeTrain = 100000;
    [Header("Changing RL")]
    public int changeRLSteps = 3000000;
    public float startLearningRate=0.4f, endLearningRate=0.05f;
    public float startInvMomentum = 10, endInvMomentum = 20;
    public bool modifyLR = true;
    [Header("Random action settings")]
    public float randomChanceStart = 1.0f;
    public float randomChanceEnd = 0.05f;
    public float randomChanceDropEpisode = 1000;
    [Header("Save and Restore")]
    public int saveStepInterval = 100000;
    public string savePath;
    public string checkpointPathNodeName = "save/Const";
    public string restoreOperationName = "save/restore_all";
    public string saveOperationName = "save/control_dependency";



    private ExperienceBuffer experienceBuffer;
    protected BrainStepMessage stepMessage;

    protected List<float> statesEpisodeHistory;
    protected List<float> rewardsEpisodeHistory;
    protected List<float> actionsEpisodeHistory;
    protected List<float> gameEndEpisodeHistory;

    protected override void Start()
    {
        base.Start();

        statesEpisodeHistory = new List<float>();
        rewardsEpisodeHistory = new List<float>();
        actionsEpisodeHistory = new List<float>();
        gameEndEpisodeHistory = new List<float>();


        experienceBuffer = new ExperienceBuffer(experienceBufferSize,
            new ExperienceBuffer.DataInfo("State", ExperienceBuffer.DataType.Float, brainsToTrain[0].brainParameters.stateSize),
            new ExperienceBuffer.DataInfo("Action", ExperienceBuffer.DataType.Float, 1),
            new ExperienceBuffer.DataInfo("Reward", ExperienceBuffer.DataType.Float, 1),
            new ExperienceBuffer.DataInfo("GameEnd", ExperienceBuffer.DataType.Float, 1)
            );
        
    }


    public override void BeforeStepTaken()
    {
        if (!training)
        {
            internalBrainsToTrain[0].whetherOverrideAction = false;
        }
        else
        {
            float ChanceOfRandom = randomChanceStart - (Mathf.Clamp01((float)Episodes / (float)randomChanceDropEpisode)) * (randomChanceStart - randomChanceEnd);
            if (UnityEngine.Random.Range(0, 1.0f) < ChanceOfRandom)
            {
                internalBrainsToTrain[0].whetherOverrideAction = true;
                float actionInt = UnityEngine.Random.Range(0, brainsToTrain[0].brainParameters.actionSize);
                internalBrainsToTrain[0].overrideAction = new Dictionary<int, float[]>();
                internalBrainsToTrain[0].overrideAction[agentToTrain.id] = new float[] { actionInt };
            }
            else
            {
                internalBrainsToTrain[0].whetherOverrideAction = false;
            }
        }
    }
    public override void OnStepTaken()
    {
        if (!training)
        {
            return;
        }
        float[] beforeState;
        if (stepMessage != null)
            beforeState = stepMessage.states[stepMessage.agents[0]].ToArray();
        else
        {
            beforeState = new float[brainsToTrain[0].brainParameters.stateSize];
        }
        //collect new message after step
        stepMessage = CollectBrainStepMessage(0);
        int agentIndex = stepMessage.agents[0];

        //add history of this step to the episode history buffer
        AddHistory(beforeState, stepMessage.rewards[agentIndex], stepMessage.actions[agentIndex], stepMessage.dones[agentIndex]);

        if (academy.done)
        {
            UpdateReplayBuffer();

            //print("done one episode");
        }

        //train
        if(TotalSteps > stepsBeforeTrain && TotalSteps % trainingStepInterval == 0)
        {
            //training
            //sample from buffer
            var samples = SampleFromBufferAll(batchSize);

            //calculate targetQs
            float[] targetQs = new float[batchSize];
            float[] maxQs = GetMaxQs((float[])samples["NextState"]);
            for (int i = 0; i < batchSize; ++i)
            {
                targetQs[i] = maxQs[i] * ((float[])samples["GameEnd"])[i] * discountFactor + ((float[])samples["Reward"])[i];
            }

            //train
            float lr, mom;
            CalculateMomentumAndLR(out mom, out lr);
            float loss = TrainBatch((float[])samples["State"], (float[])samples["Action"], targetQs, lr, mom);
        }

        //save
        if(TotalSteps > stepsBeforeTrain && TotalSteps % saveStepInterval == 0)
        {
            SaveCheckpoint();
        }
    }

    /// <summary>
    /// return the loss
    /// </summary>
    /// <param name="states"></param>
    /// <param name="actions"></param>
    /// <param name="targetQs"></param>
    /// <param name="lr"></param>
    /// <param name="mom"></param>
    /// <returns></returns>
    private float TrainBatch(float[] states, float[] actions, float[]  targetQs, float lr, float mom)
    {
        long size = targetQs.LongLength;

        int[] actionsInt = new int[actions.Length];
        for(int i = 0; i < actions.Length; ++i)
        {
            actionsInt[i] = (int)actions[i];
        }
        TFTensor inStates = TFTensor.FromBuffer(new TFShape(size, brainsToTrain[0].brainParameters.stateSize), states, 0, states.Length);
        TFTensor inTargetQ = new TFTensor(targetQs);
        TFTensor inActions = new TFTensor(actionsInt);
        
        Dictionary<string, TFTensor> feedDic = new Dictionary<string, TFTensor>();
        feedDic["input_state"] = inStates;
        feedDic["input_targetQ"] = inTargetQ;
        feedDic["input_action"] = inActions;
        feedDic["train_once/momentum"] = new TFTensor(mom);
        feedDic["train_once/learning_rate"] = new TFTensor(lr);
        TFTensor[]  resultTensors = internalBrainsToTrain[0].Run(new string[] { "output_loss" }, new string[] { "train_once" }, feedDic);
        
        return (float)(resultTensors[0].GetValue());
    }

    private float[] GetMaxQs(float[] nextState)
    {
        Dictionary<string, TFTensor> feedDic = new Dictionary<string, TFTensor>();
        feedDic["input_state"] = TFTensor.FromBuffer(new TFShape(batchSize, brainsToTrain[0].brainParameters.stateSize), nextState, 0, nextState.Length);
        TFTensor[] resultTensors = internalBrainsToTrain[0].Run(new string[] { "max_Qs" }, null, feedDic);
        float[] result = (float[])(resultTensors[0].GetValue());
        return result;
    }


    void AddHistory(float[] state, float reward, float[] action, bool gameEnd)
    {
        statesEpisodeHistory.AddRange(state);
        rewardsEpisodeHistory.Add(reward);
        actionsEpisodeHistory.AddRange(action);
        gameEndEpisodeHistory.Add(gameEnd ? 0 : 1);

    }



    Dictionary<string, Array> SampleFromBufferAll(int size)
    {
        var samples = experienceBuffer.RandomSample(size,
            Tuple.Create<string, int, string>("State", 0, "State"),
            Tuple.Create<string, int, string>("State", 1, "NextState"),
            Tuple.Create<string, int, string>("Action", 0, "Action"),
            Tuple.Create<string, int, string>("Action", 1, "NextAction"),
            Tuple.Create<string, int, string>("Reward", 0, "Reward"),
            Tuple.Create<string, int, string>("GameEnd", 0, "GameEnd"));
        return samples;
    }

    void UpdateReplayBuffer()
    {
        //print("test");
        experienceBuffer.AddExperience(Tuple.Create<string, Array>("State", statesEpisodeHistory.ToArray()),
            Tuple.Create<string, Array>("Action", actionsEpisodeHistory.ToArray()),
            Tuple.Create<string, Array>("Reward", rewardsEpisodeHistory.ToArray()),
            Tuple.Create<string, Array>("GameEnd", gameEndEpisodeHistory.ToArray())
            );

        statesEpisodeHistory.Clear();
        rewardsEpisodeHistory.Clear();
        actionsEpisodeHistory.Clear();
        gameEndEpisodeHistory.Clear();
    }


    public void SaveCheckpoint()
    {
        RunOperationWithFilePath(internalBrainsToTrain[0], checkpointPathNodeName, savePath, saveOperationName);
        print("Save to " + savePath);
    }

    public void RestoreCheckpoint()
    {
        RunOperationWithFilePath(internalBrainsToTrain[0], checkpointPathNodeName, savePath, restoreOperationName);
        print("Restore from " + savePath);
    }


    protected void CalculateMomentumAndLR(out float mom, out float lr)
    {

        float t = (Mathf.Clamp01((float)(TotalSteps - stepsBeforeTrain) / (float)changeRLSteps));

        lr = MathUtils.Interpolate(startLearningRate, endLearningRate, t, MathUtils.InterpolateMethod.Linear);
        mom = MathUtils.Interpolate(startInvMomentum, endInvMomentum, t, MathUtils.InterpolateMethod.Linear);
        mom = 1 - 1 / mom;
        if (mom < 0)
            mom = -1;
        if (modifyLR && mom >= 0 && mom < 1)
        {
            lr = lr * (1 - mom);
        }
    }
}
