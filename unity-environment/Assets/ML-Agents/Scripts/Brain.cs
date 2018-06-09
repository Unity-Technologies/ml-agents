using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System.Linq;


namespace MLAgents
{

// Class contains all necessary environment parameters
// to be defined and sent to external agent

    public enum BrainType
    {
        Player,
        Heuristic,
        External,
        Internal
    }

    public enum SpaceType
    {
        discrete,
        continuous
    };




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
        public int vectorObservationSize = 1;
        /**< \brief If continuous : The length of the float vector that represents 
         * the state
         * <br> If discrete : The number of possible values the state can take*/

        [Range(1, 50)] public int numStackedVectorObservations = 1;

        public int vectorActionSize = 1;
        /**< \brief If continuous : The length of the float vector that represents
         * the action
         * <br> If discrete : The number of possible values the action can take*/

        public resolution[] cameraResolutions;
        /**<\brief  The list of observation resolutions for the brain */

        public string[] vectorActionDescriptions;
        /**< \brief The list of strings describing what the actions correpond to */

        public SpaceType vectorActionSpaceType = SpaceType.discrete;
        /**< \brief Defines if the action is discrete or continuous */

        public SpaceType vectorObservationSpaceType = SpaceType.continuous;
        /**< \brief Defines if the state is discrete or continuous */
    }

    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
             "docs/Learning-Environment-Design-Brains.md")]
/**
 * Contains all high-level Brain logic. 
 * Add this component to an empty GameObject in your scene and drag this 
 * GameObject into your Academy to make it a child in the hierarchy.
 * Contains a set of CoreBrains, which each correspond to a different method
 * for deciding actions.
 */
    public class Brain : MonoBehaviour
    {
        private bool isInitialized = false;

        private Dictionary<Agent, AgentInfo> agentInfos =
            new Dictionary<Agent, AgentInfo>(1024);

        [Tooltip("Define state, observation, and action spaces for the Brain.")]
        /**< \brief Defines brain specific parameters such as the state size*/
        public BrainParameters brainParameters = new BrainParameters();


        /**<  \brief Defines what is the type of the brain : 
         * External / Internal / Player / Heuristic*/
        [Tooltip("Describes how the Brain will decide actions.")]
        public BrainType brainType;

        //[HideInInspector]
        ///**<  \brief Keeps track of the agents which subscribe to this brain*/
        // public Dictionary<int, Agent> agents = new Dictionary<int, Agent>();

        [SerializeField] ScriptableObject[] CoreBrains;

        /**<  \brief Reference to the current CoreBrain used by the brain*/
        public CoreBrain coreBrain;

        // Ensures the coreBrains are not dupplicated with the brains
        [SerializeField] private int instanceID;

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
                int numCoreBrains = System.Enum.GetValues(typeof(BrainType)).Length;
                CoreBrains = new ScriptableObject[numCoreBrains];
                foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
                {
                    CoreBrains[(int) bt] =
                        ScriptableObject.CreateInstance(
                            "CoreBrain" + bt.ToString());
                }

            }
            else
            {
                foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
                {
                    if ((int) bt >= CoreBrains.Length)
                        break;
                    if (CoreBrains[(int) bt] == null)
                    {
                        CoreBrains[(int) bt] =
                            ScriptableObject.CreateInstance(
                                "CoreBrain" + bt.ToString());
                    }
                }
            }

            // If the length of CoreBrains does not match the number of BrainTypes, 
            // we increase the length of CoreBrains
            if (CoreBrains.Length < System.Enum.GetValues(typeof(BrainType)).Length)
            {
                int numCoreBrains = System.Enum.GetValues(typeof(BrainType)).Length;
                ScriptableObject[] new_CoreBrains =
                    new ScriptableObject[numCoreBrains];
                foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
                {
                    if ((int) bt < CoreBrains.Length)
                    {
                        new_CoreBrains[(int) bt] = CoreBrains[(int) bt];
                    }
                    else
                    {
                        new_CoreBrains[(int) bt] =
                            ScriptableObject.CreateInstance(
                                "CoreBrain" + bt.ToString());
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
                    if (CoreBrains[(int) bt] == null)
                    {
                        CoreBrains[(int) bt] =
                            ScriptableObject.CreateInstance(
                                "CoreBrain" + bt.ToString());
                    }
                    else
                    {
                        CoreBrains[(int) bt] =
                            ScriptableObject.Instantiate(CoreBrains[(int) bt]);
                    }
                }

                instanceID = gameObject.GetInstanceID();
            }

            // The coreBrain to display is the one defined in brainType
            coreBrain = (CoreBrain) CoreBrains[(int) brainType];

            coreBrain.SetBrain(this);
        }

        /// This is called by the Academy at the start of the environemnt.
        public void InitializeBrain(Academy aca, MLAgents.Batcher brainBatcher)
        {
            UpdateCoreBrains();
            coreBrain.InitializeCoreBrain(brainBatcher);
            aca.BrainDecideAction += DecideAction;
            isInitialized = true;
        }

        public void SendState(Agent agent, AgentInfo info)
        {
            // If the brain is not active or not properly initialized, an error is
            // thrown.
            if (!gameObject.activeSelf)
            {
                throw new UnityAgentsException(
                    string.Format("Agent {0} tried to request an action " +
                                  "from brain {1} but it is not active.",
                        agent.gameObject.name, gameObject.name));
            }
            else if (!isInitialized)
            {
                throw new UnityAgentsException(
                    string.Format("Agent {0} tried to request an action " +
                                  "from brain {1} but it was not initialized.",
                        agent.gameObject.name, gameObject.name));
            }
            else
            {
                agentInfos.Add(agent, info);
            }

        }

        void DecideAction()
        {
            coreBrain.DecideAction(agentInfos);
            agentInfos.Clear();
        }
    }
}
