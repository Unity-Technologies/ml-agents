using System.Collections.Generic;
using UnityEngine.MachineLearning.InferenceEngine;
using System.Linq;
using System;
using UnityEngine;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Checks that the Tensors of an InferenceEngine are compatible with a BrainParameters
    /// object and a version of the Inference Brain. Is used to retrieve a list of strings
    /// corresponding to potential compatibility issues. If an empty IEnumerable of strings is
    /// returned when calling GetChecks, it means eather that no InferenceEngine was passed
    /// or that all checks passed.
    /// </summary>
    public static class TensorCheck
    {   
        /// <summary>
        /// Generates an IEnumerable of string corresponding to the failed compatibility checks
        /// between the InferenceEngine and the BrainParameters.
        /// </summary>
        /// <param name="engine"> The InferenceEngine going through the checks</param>
        /// <param name="inputs"> The input tensors of the InferenceEngine. Note: The checks
        /// ignore the data present in the tensors and only checks for their Name, Shape and
        /// ValueType</param>
        /// <param name="outputs">The Output tensors of the InferenceEngine. Note: The checks
        /// ignore the data present in the tensors and only checks for their Name, Shape and
        /// ValueType</param>
        /// <param name="brainParams">The BrainParameters used to evaluate the InferenceEngine
        /// checks</param>
        /// <param name="isContinuousModel"> Whether or not the InferenceEngine uses
        /// continuous control or discrete control. 0 corresponds to discrete control,
        /// 1 corresponds to continuous control and any other value signifies that the
        /// type of control could not be assessed from the InferenceEngine.</param>
        /// <param name="memoryModel">The memory size of the InferenceEngine. If the value
        /// is less or equal to zero, it means that the InferenceEngine is not using memories.
        /// </param>
        /// <param name="actionSizeModel">The action size of the InferenceEngine. If the value
        /// is less or equal to zero, it means that the action size could not be assessed from
        /// the model.
        /// </param>
        /// <returns> A IEnumerable of string corresponding to the failed checks of InferenceEngine
        /// if empty, there are no compatibility issues betweent the InferenceEngine and the
        /// BrainParameters</returns>
        public static IEnumerable<string> GetChecks(InferenceEngine engine, 
            IEnumerable<Tensor> inputs /* TODO : Remove */,
            IEnumerable<Tensor> outputs /* TODO : Remove */, 
            BrainParameters brainParams, 
            long versionModel,
            long versionBrain,
            long isContinuousModel,
            long memoryModel,
            long actionSizeModel)
        {
            if (engine == null)
            {
                return new List<string>();
            }
            var failedChecks = new List<string>();

            if (versionModel != versionBrain)
            {
                failedChecks.Add("Incompatible Version");
                return failedChecks;
            }
            if (memoryModel == -1)
            {
                failedChecks.Add("No Memory Size");
            }
            if (isContinuousModel == -1)
            {
                failedChecks.Add("Could not infer action space type from model");
            }

            if (isContinuousModel == -1)
            {
                failedChecks.Add("Could not infer action space size from model");
            }
            
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
            failedChecks.AddRange(CheckInputTensorPresence(inputs,
                brainParams,
                memoryModel,
                isContinuousModel));
            failedChecks.AddRange(CheckInputTensorShape(inputs,
                brainParams));
            failedChecks.AddRange(CheckOutputTensorPresence(outputs,
                memoryModel));
            failedChecks.AddRange(CheckOutputTensorShape(outputs,
                brainParams, actionSizeModel));
            return failedChecks;

        }
        
        private static IEnumerable<string> CheckInputTensorPresence(IEnumerable<Tensor> tensors,
            BrainParameters brainParams,
            long memory,
            long isContinuous)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((brainParams.vectorObservationSize != 0) &&
                (!tensorsNames.Contains(TensorNames.VectorObservationPlacholder)))
            {
                result.Add("The model does not contain a Vector Observation Placeholder Input. " +
                           "You must set the Vector Observation Space Size to 0.");
            }

            // If there are not enough Visual Observation Input compared to what the
            // Brain Parameters expect.
            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length;
                visObsIndex++)
            {
                if (!tensorsNames.Contains(
                    TensorNames.VisualObservationPlaceholderPrefix + visObsIndex))
                {
                    result.Add("The model does not contain a Visual Observation Placeholder " +
                               "Input for visual observation "+visObsIndex+".");
                }
            }

            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                if (!tensorsNames.Contains(TensorNames.RecurrentInPlaceholder))
                {
                    result.Add("The model does not contain a Recurrent Input Node " +
                               "but has memory_size.");
                }
            }
            
            // If the model uses discrete control but does not have an input for action masks
            if (isContinuous == 0)
            {
                if (!tensorsNames.Contains(TensorNames.ActionMaskPlaceholder))
                {
                    result.Add("The model does not contain an Action Mask but is using Discrete " +
                               "Control.");
                }
            }
            return result;
        }
        
        private static IEnumerable<string> CheckOutputTensorPresence(IEnumerable<Tensor> tensors,
            long memory)
        {
            var result = new List<string>();
            var tensorsNames = tensors.Select(x => x.Name);

            // If there is no Action Output.
            if (!tensorsNames.Contains(TensorNames.ActionOutput))
            {
                result.Add("The model does not contain an Action Output Node.");
            }
            
            // If there is no Recurrent Output but the model is Recurrent.
            if (memory > 0)
            {
                if (!tensorsNames.Contains(TensorNames.RecurrentOutput))
                {
                    result.Add("The model does not contain a Recurrent Output Node " +
                               "but has memory_size.");
                }
            }
            return result;
        }

        private static IEnumerable<string> CheckInputTensorShape(IEnumerable<Tensor> tensors, 
            BrainParameters brainParams)
        {
            var result = new List<string>();
            var tensorTester =
                new Dictionary<string, Func<Tensor, BrainParameters, string>>()
                {
                    {TensorNames.VectorObservationPlacholder, CheckVectorObsShape},
                    {TensorNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                    {TensorNames.RandomNormalEpsilonPlaceholder, ((tensor, parameters) => null)},
                    {TensorNames.ActionMaskPlaceholder, ((tensor, parameters) => null)},
                    {TensorNames.SequenceLengthPlaceholder, ((tensor, parameters) => null)},
                    {TensorNames.RecurrentInPlaceholder, ((tensor, parameters) => null)},
                };

            for (var visObsIndex = 0;
                visObsIndex < brainParams.cameraResolutions.Length; 
                visObsIndex++)
            {
                var index = visObsIndex;
                tensorTester[TensorNames.VisualObservationPlaceholderPrefix + visObsIndex] =
                    (tensor, bp) => CheckVisualObsShape(tensor, bp, index);
            }
            
            // If the model expects an input but it is not in this list
            foreach (var tensor in tensors)
            {
                if (!tensorTester.ContainsKey(tensor.Name))
                {
                    //TODO : Make this error message better
                    result.Add("No placeholder for required input : " + tensor.Name);
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
       
        private static string CheckVectorObsShape(Tensor tensor,
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
        
        private static string CheckPreviousActionShape(Tensor tensor,
            BrainParameters brainParams)
        {
            var numberActionsBp = brainParams.vectorActionSize.Length;
            var numberActionsT = tensor.Shape[1];
            if  (numberActionsBp != numberActionsT)
            {
                return string.Format(
                    "Previous Action Size of the model does not match. " +
                    "Received {0} but was expecting {2}.",
                    numberActionsBp, numberActionsT);
            }
            return null;
        }
        
        private static string CheckVisualObsShape(Tensor tensor,
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

        private static IEnumerable<string> CheckOutputTensorShape(IEnumerable<Tensor> tensors, 
            BrainParameters brainParams,
            long modelActionSize)
        {
            var result = new List<string>();
            var tensorTester =
                new Dictionary<string, Func<Tensor, BrainParameters, long, string>>();
            if (brainParams.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorTester[TensorNames.ActionOutput] = CheckContinuousActionOutputShape;
            }
            else
            {
                tensorTester[TensorNames.ActionOutput] = CheckDiscreteActionOutputShape;
            }

            // If the model expects an output but it is not in this list
            foreach (var tensor in tensors)
            {
                if (tensorTester.ContainsKey(tensor.Name))
                {
                    var tester = tensorTester[tensor.Name];
                    var error = tester.Invoke(tensor, brainParams, modelActionSize);
                    if (error != null)
                    {
                        result.Add(error);
                    }
                }
            }
            return result;
        }

        private static string CheckDiscreteActionOutputShape(Tensor tensor,
            BrainParameters brainParams,
            long modelActionSize)
        {
            var bpActionSize = brainParams.vectorActionSize.Sum();
            if  (modelActionSize != bpActionSize)
            {
                return string.Format(
                    "Action Size of the model does not match. " +
                    "The BrainParameters expect {0} but the model contains {1}.",
                    bpActionSize, modelActionSize);
            }
            return null;
        }
        
        private static string CheckContinuousActionOutputShape(Tensor tensor,
            BrainParameters brainParams,
            long modelActionSize)
        {
            var bpActionSize = brainParams.vectorActionSize[0];
            if  (modelActionSize != bpActionSize)
            {
                return string.Format(
                    "Action Size of the model does not match. " +
                    "The BrainParameters expect {0} but the model contains {1}.",
                    bpActionSize, modelActionSize);
            }
            return null;
        }
        
    }
}