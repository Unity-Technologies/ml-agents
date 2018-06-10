//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;
//
//namespace MLAgents
//{
//    /// <summary>
//    /// A Batch is a class that holds all current ScheduledPasses for a particular Archetype.
//    /// </summary>
////    public class Batch
////    {
////        public Batch()
////        {
////            inferences = new List<AgentInfo>(16);
////        }
////
////        public List<AgentInfo> inferences;
////    }
//    
//    public class MLAgentSystem
//    {
//        private static MLAgentSystem m_Instance;
//
//        public static MLAgentSystem instance
//        {
//            get { return m_Instance; }
//        }
//
//        private List<Agent> m_ActiveAgents = new List<Agent>(1024);
////        private Dictionary<System.Type, List<AgentInfo>> m_Batches = new Dictionary<System.Type, List<AgentInfo>>(1024);
////        private List<Agent> m_agents = new List<Agent>(1024);
////        private static bool m_MustRestorePlayerLoop = false;
//
//        public static void RegisterAgent(Agent archetype)
//        {
//            if (m_Instance == null)
//                return;
//
//            if (m_Instance.m_ActiveAgents.Contains(archetype))
//                return;
//            
//            m_Instance.m_ActiveAgents.Add(archetype);
//        }
//
//        public static void UnregisterArchetype(Agent archetype)
//        {
//            if (m_Instance == null)
//                return;
//
//            m_Instance.m_ActiveAgents.Remove(archetype);
//        }
//
//        public void FixedUpdateLoop()
//        {
//            // Collect / crunch loop
//            foreach (var k in m_Batches)
//            {
//                var batch = k.Value;
//
//                for (int idx = 0; idx < batch.inferences.Count; idx++)
//                {
//                    var currentInference = batch.inferences[idx];
//                    //if (currentInference.AreYouReady)
//                    //batch.collector.AddMetaDataFromArchtypeLikeRewardAndDoneFlag(currentInference.archetype);
////                    var observation = currentInference.Collect(batch.collector);
//                }
//
//                //batch.collector.CrunchData();
//                //TensorFlow.Evalutate(batch);
//            }
//
//            // Act loop
//            foreach (var k in m_Batches)
//            {
//                var batch = k.Value;
//
//                for (int idx = 0; idx < batch.inferences.Count; idx++)
//                {
//                    var currentInference = batch.inferences[idx];
//                    currentInference.Act(0);
//                }
//            }
//        }
//
//    }
//}
