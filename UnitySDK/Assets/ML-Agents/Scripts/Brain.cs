using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{

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

        public int[] vectorActionSize = new int[1]{1};
        /**< \brief If continuous : The length of the float vector that represents
         * the action
         * <br> If discrete : The number of possible values the action can take*/

        public resolution[] cameraResolutions;
        /**<\brief  The list of observation resolutions for the brain */

        public string[] vectorActionDescriptions;
        /**< \brief The list of strings describing what the actions correpond to */

        public SpaceType vectorActionSpaceType = SpaceType.discrete;
        /**< \brief Defines if the action is discrete or continuous */
    }

    /// <summary>
    /// Brain receive data from Agents through calls to SendState. The brain then updates the
    /// actions of the agents at each FixedUpdate.
    /// The Brain encapsulates the decision making process. Every Agent must be assigned a Brain,
    /// but you can use the same Brain with more than one Agent. You can also create several
    /// Brains, attach each of the Brain to one or more than one Agent.
    /// Brain assets has several important properties that you can set using the Inspector window.
    /// These properties must be appropriate for the Agents using the Brain. For example, the
    /// Vector Observation Space Size property must match the length of the feature
    /// vector created by an Agent exactly.
    /// </summary>
    public abstract class Brain : ScriptableObject
    {
        [SerializeField] public BrainParameters brainParameters;

        protected Dictionary<Agent, AgentInfo> agentInfos =
            new Dictionary<Agent, AgentInfo>(1024);

        protected Batcher brainBatcher;

        [System.NonSerialized]
        private bool _isInitialized;

        /// <summary>
        /// Sets the Batcher of the Brain. The brain will call the batcher at every step and give
        /// it the agent's data using SendBrainInfo at each DecideAction call.
        /// </summary>
        /// <param name="batcher"> The Batcher the brain will use for the current session</param>
        public void SetBatcher(Batcher batcher)
        {
            if (batcher == null)
            {
                brainBatcher = null;
            }
            else
            {
                brainBatcher = batcher;
                brainBatcher.SubscribeBrain(name);
            }
        }
        
        /// <summary>
        /// Adds the data of an agent to the current batch so it will be processed in DecideAction.
        /// </summary>
        /// <param name="agent"></param>
        /// <param name="info"></param>
        public void SendState(Agent agent, AgentInfo info)
        {
            if (!_isInitialized)
            {
                FindObjectOfType<Academy>().BrainDecideAction += DecideAction;
                Initialize();
                _isInitialized = true;
            }
            agentInfos.Add(agent, info);

        }

        protected abstract void Initialize();

        protected virtual void DecideAction()
        {
            brainBatcher?.SendBrainInfo(name, agentInfos);
        }
    }
}
