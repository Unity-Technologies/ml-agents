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

    [Header("Random action settings")]
    public float randomChanceStart = 1.0f;
    public float randomChanceEnd = 0.05f;
    public float randomChanceDropEpisode = 10000;



    private ExperienceBuffer experienceBuffer;
    private CoreBrainInternal internalBrainToTrain;

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

        internalBrainToTrain = brainsToTrain[0].coreBrain as CoreBrainInternal;
    }


    public override void BeforeStepTaken()
    {
        if (!training)
        {
            internalBrainToTrain.whetherOverrideAction = false;
        }
        else
        {
            float ChanceOfRandom = randomChanceStart - (Mathf.Clamp01((float)Episodes / (float)randomChanceDropEpisode)) * (randomChanceStart - randomChanceEnd);
            if (UnityEngine.Random.Range(0, 1.0f) < ChanceOfRandom)
            {
                internalBrainToTrain.whetherOverrideAction = true;
                float actionInt = UnityEngine.Random.Range(0, brainsToTrain[0].brainParameters.actionSize);
                internalBrainToTrain.overrideAction = new Dictionary<int, float[]>();
                internalBrainToTrain.overrideAction[agentToTrain.id] = new float[] { actionInt };
            }
            else
            {
                internalBrainToTrain.whetherOverrideAction = false;
            }
        }
    }
    public override void OnStepTaken()
    {
        if (!training)
        {
            return;
        }

        var messge = CollectBrainStepMessage(0);
        int agentIndex = messge.agents[0];

        //add history of this step to the episode history buffer
        AddHistory(messge.states[agentIndex].ToArray(), messge.rewards[agentIndex], messge.actions[agentIndex], messge.dones[agentIndex]);

        if (academy.done)
        {
            UpdateReplayBuffer();

            //print("done one episode");
        }

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
            float loss = TrainBatch((float[])samples["State"], (float[])samples["Action"], targetQs, 0.04f, 0.9f);
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
        TFTensor[]  resultTensors = internalBrainToTrain.Run(new string[] { "output_loss" }, new string[] { "train_once" }, feedDic);
        
        return (float)(resultTensors[0].GetValue());
    }

    private float[] GetMaxQs(float[] nextState)
    {
        Dictionary<string, TFTensor> feedDic = new Dictionary<string, TFTensor>();
        feedDic["input_state"] = TFTensor.FromBuffer(new TFShape(batchSize, brainsToTrain[0].brainParameters.stateSize), nextState, 0, nextState.Length);
        TFTensor[] resultTensors = internalBrainToTrain.Run(new string[] { "max_Qs" }, null, feedDic);
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
}
