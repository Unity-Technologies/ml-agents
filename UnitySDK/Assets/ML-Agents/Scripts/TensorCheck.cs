using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine;
using System.Linq;
using System;
using UnityEngine;

namespace MLAgents.InferenceBrain
{
    public class TensorCheck
    {   
        public static List<string> GetChecks(InferenceEngine engine, IEnumerable<Tensor> inputs,
            IEnumerable<Tensor> outputs, BrainParameters brainParams, long isContinuousModel,
            bool isRecurrentModel)
        {
            if (engine == null)
            {
                // TODO : Draw info directly
                return new List<string>();
            }

            var failedChecks = new List<string>();

            if (isContinuousModel == 1 &&
                brainParams.vectorActionSpaceType != SpaceType.continuous)
            {
                failedChecks.Add("Model has been trained using Continuous Control but the " +
                                 "Brain Parameters suggest Discrete Control.");
            }
            if (isContinuousModel == 0 &&
                brainParams.vectorActionSpaceType != SpaceType.discrete)
            {
                failedChecks.Add("Model has been trained using Discrete Control but the " +
                                 "Brain Parameters suggest Continuous Control.");
            }
            
            failedChecks.AddRange(CheckInputTensorShape(
                inputs,
                brainParams));
            failedChecks.AddRange(CheckInputTensorPresence(
                inputs,
                brainParams,
                isRecurrentModel,
                isContinuousModel));
            failedChecks.AddRange(CheckOutputTensorPresence(
                outputs,
                isRecurrentModel));
            return failedChecks;

        }
        
        private static List<string> CheckInputTensorPresence(
            IEnumerable<Tensor> tensors,
            BrainParameters brainParams,
            bool isRecurrent,
            long isContinuous)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((brainParams.vectorObservationSize != 0) &&
                (!tensorsNames.Contains(NodeNames.VectorObservationPlacholder)))
            {
                result.Add("The model does not contain a Vector Observation Placeholder Input. " +
                           "You must set the Vector Observation Space Size to 0.");
            }

            // If there is too little Visual Observation Input compared to what the
            // Brain Parameters expect.
            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length;
                visObsIndex++)
            {
                if (!tensorsNames.Contains(
                    NodeNames.VisualObservationPlaceholderPrefix + visObsIndex))
                {
                    result.Add("The model does not contain a Visual Observation Placeholder " +
                               "Input for visual observation "+visObsIndex+".");
                }
            }

            if (isRecurrent)
            {
                if (!tensorsNames.Contains(NodeNames.RecurrentInPlaceholder))
                {
                    result.Add("The model does not contain a Recurrent Input Node " +
                               "but has memory_size.");
                }
            }

            if (isContinuous == 0)
            {
                if (!tensorsNames.Contains(NodeNames.ActionMaskPlaceholder))
                {
                    result.Add("The model does not contain an Action Mask but is using Discrete " +
                               "Control.");
                }
            }
            // Epsilon placeholder are optional
            
            return result;
        }
        
        private static List<string> CheckOutputTensorPresence(
            IEnumerable<Tensor> tensors,
            bool isRecurrent)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Action Output.
            if (!tensorsNames.Contains(NodeNames.ActionOutput))
            {
                result.Add("The model does not contain an Action Output Node.");
            }
            
            if (isRecurrent)
            {
                // If there is no Recurrent Output but the model is Recurrent.
                if (!tensorsNames.Contains(NodeNames.RecurrentOutOutput))
                {
                    result.Add("The model does not contain a Recurrent Output Node " +
                               "but has memory_size.");
                }
            }
            
            // Value estimates are optional

            return result;
        }

        private static List<string> CheckInputTensorShape(
            IEnumerable<Tensor> tensors, 
            BrainParameters brainParams)
        {
            var result = new List<string>();
 
            var tensorTester =
                new Dictionary<string, Func<Tensor, BrainParameters, string>>()
                {
                    {NodeNames.VectorObservationPlacholder, CheckVectorObsShape},
                    {NodeNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                    {NodeNames.RandomNormalEpsilonPlaceholder, ((tensor, parameters) => null)},
                    {NodeNames.ActionMaskPlaceholder, ((tensor, parameters) => null)},
                    {NodeNames.SequenceLengthPlaceholder, ((tensor, parameters) => null)},
                    {NodeNames.RecurrentInPlaceholder, ((tensor, parameters) => null)},
                };

            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length; 
                visObsIndex++)
            {
                var index = visObsIndex;
                tensorTester[NodeNames.VisualObservationPlaceholderPrefix + visObsIndex] =
                    (tensor, bp) => CheckVisualObsShape(tensor, bp, index);
            }
            
            // If the model expects an input but it is not in this list
            foreach (var tensor in tensors)
            {
                if (!tensorTester.ContainsKey(tensor.Name))
                {
                    result.Add("No placeholder for input : " + tensor.Name);
                }
                else
                {
                    var tester = tensorTester[tensor.Name];
                    var error = tester.Invoke(tensor, brainParams);
                    if (error != null)
                    {
                        result.Add(error);
                    }
                }
            }
            return result;
        }
       
        private static string CheckVectorObsShape(
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
        
        private static string CheckPreviousActionShape(
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
        
        private static string CheckVisualObsShape(
            Tensor tensor,
            BrainParameters brainParams,
            int visObsIndex)
        {
            
            var resolutionBp = brainParams.cameraResolutions[visObsIndex];
            var widthBp = resolutionBp.width;
            var heightBp = resolutionBp.height;
            var pixelBp = resolutionBp.blackAndWhite ? 1 : 3;
            var widthT = tensor.Shape[1];
            var heightT = tensor.Shape[2];
            var pixelT = tensor.Shape[3];
            if  ((widthBp != widthT) || (heightBp != heightT) || (pixelBp != pixelT))
            {
                return string.Format(
                    "The visual Observation {0} of the model does not match. " +
                    "Received Tensor of shape [?x{1}x{2}x{3}] but was expecting [?x{4}x{5}x{6}].",
                    visObsIndex, widthBp, heightBp, pixelBp, widthT, heightT, pixelT);
            }
            return null;
        }
        
    }
}