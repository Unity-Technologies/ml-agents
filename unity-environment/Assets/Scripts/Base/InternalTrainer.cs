using System.Collections;
using System.Collections.Generic;
using UnityEngine;






public class InternalTrainer : MonoBehaviour {

    //the academy that needs to be trained. The acadamy needs to have internalbrains
    public Academy academy;
    public List<Brain> brainsToTrain;
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



    
}
