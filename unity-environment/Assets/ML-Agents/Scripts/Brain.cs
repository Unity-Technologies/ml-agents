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
    public void InitializeBrain(Academy aca, Communicator communicator)
    {
        UpdateCoreBrains();
        coreBrain.InitializeCoreBrain(communicator);
        aca.OnDecideAction += DecideAction;
    }

    public void SendState(Agent agent, AgentInfo info)
    {
        agentInfos.Add(agent, info);

    }

    void DecideAction()
    {
        coreBrain.DecideAction(agentInfos);
        agentInfos.Clear();

    }

    public bool IsBroadcasting(){
        return coreBrain.IsBroadcasting();
    }

}