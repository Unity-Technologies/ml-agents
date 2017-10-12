using System.Collections;
using System.Collections.Generic;
using UnityEngine;






public class InternalTrainer : MonoBehaviour {

    //the academy that needs to be trained. The acadamy needs to have internalbrains
    public Academy academy;
    public List<Brain> brainsToTrain;


    public bool training = false;
    
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


    // Use this for initialization
    void Start () {
        

    }
	
	// Update is called once per frame
	void Update () {
		
	}

    private void FixedUpdate()
    {
        if(academy!= null && training)
        {
            academy.RunStepManual();
            if (academy.done)
            {
                academy.RunResetManual();
            }
            
        }
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
