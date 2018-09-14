using UnityEngine.MachineLearning.InferenceEngine;
using System.Linq;
using System.Collections.Generic;

namespace MLAgents.CoreInternalBrain
{
    public class NodeTester
    {
//        public NodeTester(
//            InferenceEngine inferenceEngine, 
//            BrainParameters brainParameters,
//            int kApiNumber)
//        {
//            
//        }
        
        
        // Put this into a separate class
        public static List<string> TestInputTensorShape(
            Tensor[] tensors, 
            BrainParameters brainParams,
            NodeNames nodeNames)
        {
            List<string> result = new List<string>();
            foreach (var tensor in tensors)
            {
                if (tensor.Name == nodeNames.VectorObservationPlacholder)
                {
                    result.Add(TestVectorObsShape(tensor, brainParams));
                }
                else if (tensor.Name == nodeNames.PreviousActionPlaceholder)
                {
                    result.Add(TestPreviousActionShape(tensor, brainParams));
                }
                
            }
            return result.Where(x => x!=null).ToList();
        }

        private static string TestVectorObsShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            var vecObsSizeBp = brainParams.vectorObservationSize;
            var numStackedVector = brainParams.numStackedVectorObservations;
            var totalVecObsSizeT = tensor.Shape[1];
            if (vecObsSizeBp * numStackedVector != totalVecObsSizeT)
            {
                return string.Format(
                    "Vector Observation Size of the model does not match. " +
                    "Received {0} x {1} but was expecting {2}.",
                    vecObsSizeBp, numStackedVector, totalVecObsSizeT);
            }
            return null;
        }
        
        private static string TestPreviousActionShape(
            Tensor tensor,
            BrainParameters brainParams)
        {
            var numberActionsBp = brainParams.vectorActionSize.Length;
            var numberActionsT = tensor.Shape[1];
            if  (numberActionsBp != numberActionsT)
            {
                return string.Format(
                    "Action Size of the model does not match. " +
                    "Received {0} but was expecting {2}.",
                    numberActionsBp, numberActionsT);
            }
            return null;
        }

    }
}