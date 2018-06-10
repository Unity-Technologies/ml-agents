using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
//    public enum PolicyType
//    {
//        Learned,
//        Human,
//        Scripted
//    }
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
    
    public abstract class Brain : ScriptableObject
    {
        [SerializeField] public BrainParameters brainParameters;
        
        protected Dictionary<Agent, AgentInfo> agentInfo =
            new Dictionary<Agent, AgentInfo>(1024);
        
//        public PolicyType m_Policy;
        public string m_ArchetypeName;

        protected MLAgents.Batcher brainBatcher;

        public bool isExternal;

        public virtual void InitializeBrain(Academy aca, Batcher batcher, bool training)
        {
            isExternal = isExternal && training;
            aca.BrainDecideAction += DecideAction;
            if (batcher == null)
            {
                brainBatcher = null;
            }
            else
            {
                brainBatcher = batcher;
                brainBatcher.SubscribeBrain(this.name);
            }
        }
        /*TODO : Rename, split and implement here. Initialization should be done at the first
         send state call and subscribe batcher should be called by the academy.*/
        
        public void SendState(Agent agent, AgentInfo info)
        {
            agentInfo.Add(agent, info);

        }
        
        protected abstract void DecideAction();
    }
    
    

}