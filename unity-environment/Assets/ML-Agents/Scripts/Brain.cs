using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System.Linq;



// Class contains all necessary environment parameters
// to be defined and sent to external agent
#if ENABLE_TENSORFLOW
public enum BrainType
{
    Player,
    Heuristic,
    External,
    Internal
}

#else
public enum BrainType
{
    Player,
    Heuristic,
    External,
}
#endif




public enum StateType
{
    discrete,
    continuous
}
;




/** Only need to be modified in the brain's inpector.
 * Defines what is the resolution of the camera
*/
[System.Serializable]
public struct resolution
{
    public int width;
    /**< \brief The width of the observation in pixels */
    public int height;
    /**< \brief The height of the observation in pixels */
    public bool blackAndWhite;
    /**< \brief If true, the image will be in black and white. 
     * If false, it will be in colors RGB */
}

/** Should be modified via the Editor Inspector.
 * Defines brain-specific parameters
*/
[System.Serializable]
public class BrainParameters
{
    [Tooltip("Length of state vector for brain (In Continuous state space)." +
             "Or number of possible values (in Discrete state space).")]
    public int stateSize = 1;
    /**< \brief If continuous : The length of the float vector that represents 
     * the state
     * <br> If discrete : The number of possible values the state can take*/
    [Tooltip("Number of states that will be staked before beeing fed to the neural network.")]
    [Range(1, 10)]
    public int stackedStates = 1;

    [Tooltip("Length of action vector for brain (In Continuous state space)." +
             "Or number of possible values (in Discrete action space).")]
    public int actionSize = 1;
    /**< \brief If continuous : The length of the float vector that represents the action
     * <br> If discrete : The number of possible values the action can take*/
    [Tooltip("Length of memory vector for brain. Used with Recurrent networks.")]
    public int memorySize = 0;
    /**< \brief The length of the float vector that holds the memory for the agent */
    [Tooltip("Describes height, width, and whether to greyscale visual observations for the Brain.")]
    public resolution[] cameraResolutions;
    /**<\brief  The list of observation resolutions for the brain */
    [Tooltip("A list of strings used to name the available actions for the Brain.")]
    public string[] actionDescriptions;
    /**< \brief The list of strings describing what the actions correpond to */
    [Tooltip("Corresponds to whether state vector contains a single integer (Discrete) " +
             "or a series of real-valued floats (Continuous).")]
    public StateType actionSpaceType = StateType.discrete;
    /**< \brief Defines if the action is discrete or continuous */
    [Tooltip("Corresponds to whether action vector contains a single integer (Discrete)" +
             " or a series of real-valued floats (Continuous).")]
    public StateType stateSpaceType = StateType.continuous;
    /**< \brief Defines if the state is discrete or continuous */
}

[HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Agents-Editor-Interface.md#brain")]
/**
 * Contains all high-level Brain logic. 
 * Add this component to an empty GameObject in your scene and drag this 
 * GameObject into your Academy to make it a child in the hierarchy.
 * Contains a set of CoreBrains, which each correspond to a different method
 * for deciding actions.
 */
public class Brain : MonoBehaviour
{
    // Current agent info

    private Dictionary<Agent, AgentInfo> agentInfos = new Dictionary<Agent, AgentInfo>(1024);

    //private int b;


    [Tooltip("Define state, observation, and action spaces for the Brain.")]
    /**< \brief Defines brain specific parameters such as the state size*/
    public BrainParameters brainParameters = new BrainParameters();


    /**<  \brief Defines what is the type of the brain : 
     * External / Internal / Player / Heuristic*/
    [Tooltip("Describes how the Brain will decide actions.")]
    public BrainType brainType;

    //[HideInInspector]
    ///**<  \brief Keeps track of the agents which subscribe to this brain*/
    //public Dictionary<int, Agent> agents = new Dictionary<int, Agent>();

    [SerializeField]
    ScriptableObject[] CoreBrains;

    /**<  \brief Reference to the current CoreBrain used by the brain*/
    public CoreBrain coreBrain;

    //Ensures the coreBrains are not dupplicated with the brains
    [SerializeField]
    private int instanceID;

    /// Ensures the brain has an up to date array of coreBrains
    /** Is called when the inspector is modified and into InitializeBrain. 
	 * If the brain gameObject was just created, it generates a list of 
	 * coreBrains (one for each brainType) */
    public void UpdateCoreBrains()
    {

        // If CoreBrains is null, this means the Brain object was just 
        // instanciated and we create instances of each CoreBrain
        if (CoreBrains == null)
        {
            CoreBrains = new ScriptableObject[System.Enum.GetValues(typeof(BrainType)).Length];
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
            }

        }
        else
        {
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                if ((int)bt >= CoreBrains.Length)
                    break;
                if (CoreBrains[(int)bt] == null)
                {
                    CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
                }
            }
        }

        // If the length of CoreBrains does not match the number of BrainTypes, 
        // we increase the length of CoreBrains
        if (CoreBrains.Length < System.Enum.GetValues(typeof(BrainType)).Length)
        {
            ScriptableObject[] new_CoreBrains = new ScriptableObject[System.Enum.GetValues(typeof(BrainType)).Length];
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                if ((int)bt < CoreBrains.Length)
                {
                    new_CoreBrains[(int)bt] = CoreBrains[(int)bt];
                }
                else
                {
                    new_CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
                }
            }
            CoreBrains = new_CoreBrains;
        }

        // If the stored instanceID does not match the current instanceID, 
        // this means that the Brain GameObject was duplicated, and
        // we need to make a new copy of each CoreBrain
        if (instanceID != gameObject.GetInstanceID())
        {
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                if (CoreBrains[(int)bt] == null)
                {
                    CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
                }
                else
                {
                    CoreBrains[(int)bt] = ScriptableObject.Instantiate(CoreBrains[(int)bt]);
                }
            }
            instanceID = gameObject.GetInstanceID();
        }

        // The coreBrain to display is the one defined in brainType
        coreBrain = (CoreBrain)CoreBrains[(int)brainType];

        coreBrain.SetBrain(this);
    }

    /// This is called by the Academy at the start of the environemnt.
    public void InitializeBrain()
    {
        UpdateCoreBrains();
        coreBrain.InitializeCoreBrain();

    }


<<<<<<< HEAD
    public void SendState(Agent agent, AgentInfo info)
=======
    public void CollectEverything()
    {
        currentStates.Clear();
        currentCameras.Clear();
        currentRewards.Clear();
        currentDones.Clear();
        currentMaxes.Clear();
        currentActions.Clear();
        currentMemories.Clear();

        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            idAgent.Value.SetCumulativeReward();
            List<float> states = idAgent.Value.ClearAndCollectState();
            if ((states.Count != brainParameters.stateSize * brainParameters.stackedStates) && (brainParameters.stateSpaceType == StateType.continuous))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting {1} continuous states but received {2}.", idAgent.Value.gameObject.name, brainParameters.stateSize, states.Count));
            }
            if ((states.Count != brainParameters.stackedStates) && (brainParameters.stateSpaceType == StateType.discrete))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting 1 discrete states but received {1}.", idAgent.Value.gameObject.name, states.Count));
            }

            List<Camera> observations = idAgent.Value.observations;
            if (observations.Count < brainParameters.cameraResolutions.Count())
            {
                throw new UnityAgentsException(string.Format(@"The number of observations does not match for agent {0}:
    Was expecting at least {1} observation but received {2}.", idAgent.Value.gameObject.name, brainParameters.cameraResolutions.Count(), observations.Count));
            }

            currentStates.Add(idAgent.Key, states);
            currentCameras.Add(idAgent.Key, observations);
            currentRewards.Add(idAgent.Key, idAgent.Value.reward);
            currentDones.Add(idAgent.Key, idAgent.Value.done);
            currentMaxes.Add(idAgent.Key, idAgent.Value.maxStepReached);
            currentActions.Add(idAgent.Key, idAgent.Value.agentStoredAction);
            currentMemories.Add(idAgent.Key, idAgent.Value.memory);
        }
    }


    /// Collects the states of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> state}
    public Dictionary<int, List<float>> CollectStates()
    {
        currentStates.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            idAgent.Value.SetCumulativeReward();
            List<float> states = idAgent.Value.ClearAndCollectState();
            if ((states.Count != brainParameters.stateSize * brainParameters.stackedStates) && (brainParameters.stateSpaceType == StateType.continuous))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting {1} continuous states but received {2}.", idAgent.Value.gameObject.name, brainParameters.stateSize * brainParameters.stackedStates, states.Count));
            }
            if ((states.Count != brainParameters.stackedStates) && (brainParameters.stateSpaceType == StateType.discrete))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting {1} discrete states but received {2}.", idAgent.Value.gameObject.name, brainParameters.stackedStates, states.Count));
            }
            currentStates.Add(idAgent.Key, states);
        }
        return currentStates;
    }

    /// Collects the observations of all the agents which subscribe to this 
    /// brain and returns a dictionary {id -> Camera}
    public Dictionary<int, List<Camera>> CollectObservations()
    {
        currentCameras.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            List<Camera> observations = idAgent.Value.observations;
            if (observations.Count < brainParameters.cameraResolutions.Count())
            {
                throw new UnityAgentsException(string.Format(@"The number of observations does not match for agent {0}:
	Was expecting at least {1} observation but received {2}.", idAgent.Value.gameObject.name, brainParameters.cameraResolutions.Count(), observations.Count));
            }
            currentCameras.Add(idAgent.Key, observations);
        }
        return currentCameras;

    }

    /// Collects the rewards of all the agents which subscribe to this brain
    /// and returns a dictionary {id -> reward}
    public Dictionary<int, float> CollectRewards()
    {
        currentRewards.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            currentRewards.Add(idAgent.Key, idAgent.Value.reward);
        }
        return currentRewards;
    }

    /// Collects the done flag of all the agents which subscribe to this brain
    ///  and returns a dictionary {id -> done}
    public Dictionary<int, bool> CollectDones()
    {
        currentDones.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            currentDones.Add(idAgent.Key, idAgent.Value.done);
        }
        return currentDones;
    }

    /// Collects the done flag of all the agents which subscribe to this brain
    ///  and returns a dictionary {id -> done}
    public Dictionary<int, bool> CollectMaxes()
    {
        currentMaxes.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            currentMaxes.Add(idAgent.Key, idAgent.Value.maxStepReached);
        }
        return currentMaxes;
    }


    /// Collects the actions of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> action}
    public Dictionary<int, float[]> CollectActions()
    {
        currentActions.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            currentActions.Add(idAgent.Key, idAgent.Value.agentStoredAction);
        }
        return currentActions;
    }

    /// Collects the memories of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> memories}
    public Dictionary<int, float[]> CollectMemories()
    {
        currentMemories.Clear();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            currentMemories.Add(idAgent.Key, idAgent.Value.memory);
        }
        return currentMemories;
    }

    /// Takes a dictionary {id -> memories} and sends the memories to the 
    /// corresponding agents
    public void SendMemories(Dictionary<int, float[]> memories)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            idAgent.Value.memory = memories[idAgent.Key];
        }
    }

    /// Takes a dictionary {id -> actions} and sends the actions to the 
    /// corresponding agents
    public void SendActions(Dictionary<int, float[]> actions)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            //Add a check here to see if the component was destroyed ?
            idAgent.Value.UpdateAction(actions[idAgent.Key]);
        }
    }

    /// Takes a dictionary {id -> values} and sends the values to the 
    /// corresponding agents
    public void SendValues(Dictionary<int, float> values)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            //Add a check here to see if the component was destroyed ?
            idAgent.Value.value = values[idAgent.Key];
        }
    }

    ///Sets all the agents which subscribe to the brain to done
    public void SendDone()
>>>>>>> origin/development-0.3
    {
        agentInfos.Add(agent, info);

    }

    public void DecideAction()
    {
        coreBrain.DecideAction(agentInfos);
        agentInfos.Clear();

    }

}