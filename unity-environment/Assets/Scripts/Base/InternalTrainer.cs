using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;
using System.Text;

public class InternalTrainer : MonoBehaviour {

    //the academy that needs to be trained. The acadamy needs to have internalbrains
    public Academy academy;
    public List<Brain> brainsToTrain;
    protected List<CoreBrainInternal> internalBrainsToTrain;
    public Agent agentToTrain;
    public bool running = true;
    
    public int TotalSteps { get; protected set; }
    public int StepsFromLastReset { get; protected set; }
    public int Episodes { get; protected set; }
    protected virtual void Start()
    {
        TotalSteps = 0;
        StepsFromLastReset = 0;
        Episodes = 0;

        internalBrainsToTrain = new List<CoreBrainInternal>();
        foreach(var b in brainsToTrain)
        {
            internalBrainsToTrain.Add(b.coreBrain as CoreBrainInternal);
        }
    }

    public class BrainStepMessage
    {
        public string brain_name { get; set; }
        public List<int> agents { get; set; }           //game object id of those agents
        public Dictionary<int, List<float>> states { get; set; }
        public Dictionary<int, float> rewards { get; set; }
        public Dictionary<int, float[]> actions { get; set; }
        public Dictionary<int, float[]> memories { get; set; }
        public Dictionary<int, bool> dones { get; set; }

        public List<resolution> cameraResolutions { get; set; } //observations' camera resolutions of this brain
        public Dictionary<int, List<byte[]>> collectedObservations { get; set; }
    }



    private void FixedUpdate()
    {
        if(academy!= null && running)
        {
            BeforeStepTaken();
            academy.RunStepManual();
            TotalSteps ++;
            StepsFromLastReset ++;
            OnStepTaken();
            if (academy.done)
            {
                academy.RunResetManual();
                StepsFromLastReset = 0;
                Episodes++;
            }
        }
    }


    /// <summary>
    /// override this to run the training
    /// </summary>
    public virtual void OnStepTaken()
    {

    }

    /// <summary>
    /// override this to run tstuff before step is taken
    /// </summary>
    public virtual void BeforeStepTaken()
    {

    }

    public BrainStepMessage CollectBrainStepMessage(int brainIndex = 0)
    {
        Debug.Assert(brainIndex < brainsToTrain.Count && brainIndex >=0, "brain index out of bound");

        Brain brain = brainsToTrain[brainIndex];
        
        Dictionary<int, List<Camera>> collectedObservations = brain.CollectObservations();

        Dictionary<int, List<byte[]>> byteObservaations = new Dictionary<int, List<byte[]>>();

        int i = 0;
        foreach (resolution res in brain.brainParameters.cameraResolutions)
        {
            foreach (int id in brain.agents.Keys)
            {
                byteObservaations[id].Add(
                    ExternalCommunicator.TexToByteArray(
                        brain.ObservationToTex(collectedObservations[id][i], res.width, res.height)
                    )
                );
            }
            i++;
        }



        BrainStepMessage message = new BrainStepMessage()
        {
            brain_name = brain.gameObject.name,
            agents = new List<int>(brain.agents.Keys),
            states = brain.CollectStates(),
            rewards = brain.CollectRewards(),
            actions = brain.CollectActions(),
            memories = brain.CollectMemories(),
            dones = brain.CollectDones(),
            cameraResolutions = new List<resolution>(brain.brainParameters.cameraResolutions),
            collectedObservations = byteObservaations
        };

        return message;

    }
    

    public static void RunOperationWithFilePath(CoreBrainInternal internalBrain, string fileNameNodeName, string filePath, string operationName)
    {
        TFTensor tensorStringPath = BuildPathTensor(filePath);
        Dictionary<string, TFTensor> feedDic = new Dictionary<string, TFTensor>();
        feedDic[fileNameNodeName] = tensorStringPath;
        internalBrain.Run(null, new string[] { operationName }, feedDic);
    }

    /// <summary>
    /// A helper function to generate a Tensor
    /// </summary>
    /// <param name="filenamenodeName">The operation name in the graph that specifies the checkpoint model file name </param>
    /// <param name="filePathString">A string of the subpath of the check point model file</param>
    /// <returns>TFTensor that contains the full path of the input string</returns>
    public static TFTensor BuildPathTensor(string filePathString)
    {

        //get the full path of the file
        string fullFilePath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), filePathString);

        fullFilePath = System.IO.Path.GetFullPath(filePathString);
        //create a tensor from the path string
        TFTensor tensorStringPath = TFTensor.CreateString(Encoding.ASCII.GetBytes(fullFilePath));
        return tensorStringPath;
    }

}
