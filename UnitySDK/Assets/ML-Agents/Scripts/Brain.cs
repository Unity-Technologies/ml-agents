using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System.Linq;
using UnityEditor;


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

/**
 * Contains all high-level Brain logic. 
 * Add this component to an empty GameObject in your scene and drag this 
 * GameObject into your Academy to make it a child in the hierarchy.
 * Contains a set of CoreBrains, which each correspond to a different method
 * for deciding actions.
 */
    public abstract class Brain : ScriptableObject
    {
        [SerializeField] public BrainParameters brainParameters;

        protected Dictionary<Agent, AgentInfo> agentInfos =
            new Dictionary<Agent, AgentInfo>(1024);

        protected Batcher brainBatcher;

        [System.NonSerialized]
        private bool _isInitialized;

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
        /*TODO : Rename, split and implement here. Initialization should be done at the first
         send state call and subscribe batcher should be called by the academy.*/
        
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
