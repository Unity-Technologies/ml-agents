using System;
using System.Collections.Generic;
using System.Linq;
using TransformsAI.MicroMLAgents.Actuators;
using TransformsAI.MicroMLAgents.Sensors;
using Unity.Barracuda;

namespace TransformsAI.MicroMLAgents.Inference
{
    /// <summary>
    /// Prepares the Tensors for the Learning Brain and exposes a list of failed checks if Model
    /// and BrainParameters are incompatible.
    /// </summary>
    internal class BarracudaModelParamLoader
    {

        internal enum ModelApiVersion
        {
            /// <summary>
            /// ML-Agents model version for versions 1.x.y
            /// The observations are split between vector and visual observations
            /// There are legacy action outputs for discrete and continuous actions
            /// LSTM inputs and outputs are handled by Barracuda
            /// </summary>
            MLAgents1_0 = 2,

            /// <summary>
            /// All observations are treated the same and named obs_{i} with i being
            /// the sensor index
            /// Legacy "action" output is no longer present
            /// LSTM inputs and outputs are treated like regular inputs and outputs
            /// and no longer managed by Barracuda
            /// </summary>
            MLAgents2_0 = 3,
            MinSupportedVersion = MLAgents2_0,
            MaxSupportedVersion = MLAgents2_0
        }

        internal class FailedCheck
        {
            public enum CheckTypeEnum
            {
                Info = 0,
                Warning = 1,
                Error = 2
            }
            public CheckTypeEnum CheckType;
            public string Message;
            public static FailedCheck Info(string message)
            {
                return new FailedCheck { CheckType = CheckTypeEnum.Info, Message = message };
            }
            public static FailedCheck Warning(string message)
            {
                return new FailedCheck { CheckType = CheckTypeEnum.Warning, Message = message };
            }
            public static FailedCheck Error(string message)
            {
                return new FailedCheck { CheckType = CheckTypeEnum.Error, Message = message };
            }
        }

        /// <summary>
        /// Checks that a model has the appropriate version.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <returns>A FailedCheck containing the error message if the version of the model does not mach, else null</returns>
        public static FailedCheck CheckModelVersion(Model model)
        {
            var modelApiVersion = model.GetVersion();
            if (modelApiVersion < (int)ModelApiVersion.MinSupportedVersion)
            {
                return FailedCheck.Error(
                    "Model was trained with a older version of the trainer than is supported. " +
                    "Either retrain with an newer trainer, or use an older version of com.unity.ml-agents.\n" +
                    $"Model version: {modelApiVersion} Minimum supported version: {(int)ModelApiVersion.MinSupportedVersion}"
                );
            }

            if (modelApiVersion > (int)ModelApiVersion.MaxSupportedVersion)
            {
                return FailedCheck.Error(
                    "Model was trained with a newer version of the trainer than is supported. " +
                    "Either retrain with an older trainer, or update to a newer version of com.unity.ml-agents.\n" +
                    $"Model version: {modelApiVersion}  Maximum supported version: {(int)ModelApiVersion.MaxSupportedVersion}"
                );
            }

            var memorySize = (int)model.GetTensorByName(TensorNames.MemorySize)[0];

            if (modelApiVersion == (int)ModelApiVersion.MLAgents1_0 && memorySize > 0)
            {
                // This block is to make sure that models that are trained with MLAgents version 1.x and have
                // an LSTM (i.e. use the barracuda _c and _h inputs and outputs) will not work with MLAgents version
                // 2.x. This is because Barracuda version 2.x will eventually drop support for the _c and _h inputs
                // and only ML-Agents 2.x models will be compatible.
                return FailedCheck.Error(
                    "Models from com.unity.ml-agents 1.x that use recurrent neural networks are not supported in newer versions. " +
                    "Either retrain with an newer trainer, or use an older version of com.unity.ml-agents.\n"
                );
            }
            return null;

        }



        /// <summary>
        /// Factory for the ModelParamLoader : Creates a ModelParamLoader and runs the checks
        /// on it.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="sensors">Attached sensor components</param>
        /// <param name="actuatorComponents">Attached actuator components</param>
        /// <param name="observableAttributeTotalSize">Sum of the sizes of all ObservableAttributes.</param>
        /// <param name="behaviorType">BehaviorType or the Agent to check.</param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        /// <returns>A IEnumerable of the checks that failed</returns>
        public static IEnumerable<FailedCheck> CheckModel(
            Model model,
            ActionSpec actionSpec,
            ObservationSpec[] sensors,
            bool deterministicInference = false
            )
        {
            List<FailedCheck> failedModelChecks = new List<FailedCheck>();
            if (model == null)
            {
                var errorMsg = "There is no model for this Brain; cannot run inference. ";
                failedModelChecks.Add(FailedCheck.Info(errorMsg));
                return failedModelChecks;
            }

            var hasExpectedTensors = model.CheckExpectedTensors(failedModelChecks, deterministicInference);
            if (!hasExpectedTensors)
            {
                return failedModelChecks;
            }

            var modelApiVersion = model.GetVersion();
            var versionCheck = CheckModelVersion(model);
            if (versionCheck != null)
            {
                failedModelChecks.Add(versionCheck);
            }

            var memorySize = (int)model.GetTensorByName(TensorNames.MemorySize)[0];
            if (memorySize == -1)
            {
                failedModelChecks.Add(FailedCheck.Warning($"Missing node in the model provided : {TensorNames.MemorySize}"
                    ));
                return failedModelChecks;
            }

            if (modelApiVersion != (int)ModelApiVersion.MLAgents2_0)
                throw new Exception("Unsupported MLAgents Version");

            failedModelChecks.AddRange(
                CheckInputTensorPresence(model, memorySize, sensors, deterministicInference)
            );
            failedModelChecks.AddRange(
                CheckInputTensorShape(model,actionSpec, sensors)
            );

            failedModelChecks.AddRange(
                CheckOutputTensorShape(model, actionSpec)
            );

            failedModelChecks.AddRange(
                CheckOutputTensorPresence(model, memorySize, deterministicInference)
            );
            return failedModelChecks;
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="memory">
        /// The memory size that the model is expecting.
        /// </param>
        /// <param name="sensors">Array of attached sensor components</param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// Deterministic. </param>
        /// <returns>
        /// A IEnumerable of the checks that failed
        /// </returns>
        static IEnumerable<FailedCheck> CheckInputTensorPresence(
            Model model,
            int memory,
            ObservationSpec[] sensors,
            bool deterministicInference = false
        )
        {
            var failedModelChecks = new List<FailedCheck>();
            var tensorsNames = model.GetInputNames();
            for (var sensorIndex = 0; sensorIndex < sensors.Length; sensorIndex++)
            {
                if (!tensorsNames.Contains(
                    TensorNames.GetObservationName(sensorIndex)))
                {
                    var sensor = sensors[sensorIndex];
                    failedModelChecks.Add(
                        FailedCheck.Warning("The model does not contain an Observation Placeholder Input " +
                            $"for sensor component {sensorIndex}.")
                        );
                }
            }

            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                var modelVersion = model.GetVersion();
                if (tensorsNames.All(x => x != TensorNames.RecurrentInPlaceholder))
                {
                    failedModelChecks.Add(
                            FailedCheck.Warning("The model does not contain a Recurrent Input Node but has memory_size.")
                            );
                }
            }

            // If the model uses discrete control but does not have an input for action masks
            if (model.HasDiscreteOutputs(deterministicInference))
            {
                if (!tensorsNames.Contains(TensorNames.ActionMaskPlaceholder))
                {
                    failedModelChecks.Add(
                        FailedCheck.Warning("The model does not contain an Action Mask but is using Discrete Control.")
                        );
                }
            }
            return failedModelChecks;
        }

        /// <summary>
        /// Generates failed checks that correspond to outputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="memory">The memory size that the model is expecting/</param>
        /// <param name="deterministicInference"> Inference only: set to true if the action selection from model should be
        /// deterministic. </param>
        /// <returns>
        /// A IEnumerable of the checks that failed
        /// </returns>
        static IEnumerable<FailedCheck> CheckOutputTensorPresence(Model model, int memory, bool deterministicInference = false)
        {
            var failedModelChecks = new List<FailedCheck>();

            // If there is no Recurrent Output but the model is Recurrent.
            if (memory > 0)
            {
                var allOutputs = model.GetOutputNames(deterministicInference).ToList();
                if (allOutputs.All(x => x != TensorNames.RecurrentOutput))
                {
                    failedModelChecks.Add(
                        FailedCheck.Warning("The model does not contain a Recurrent Output Node but has memory_size.")
                        );
                }

            }
            return failedModelChecks;
        }

        /// <summary>
        /// Checks that the shape of the visual observation input placeholder is the same as the corresponding sensor.
        /// </summary>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <param name="sensor">The sensor that produces the visual observation.</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckVisualObsShape(
            TensorProxy tensorProxy, ObservationSpec sensor)
        {
            var shape = sensor.Shape;
            var heightBp = shape[0];
            var widthBp = shape[1];
            var pixelBp = shape[2];
            var heightT = tensorProxy.Height;
            var widthT = tensorProxy.Width;
            var pixelT = tensorProxy.Channels;
            if ((widthBp != widthT) || (heightBp != heightT) || (pixelBp != pixelT))
            {
                return FailedCheck.Warning($"The visual Observation of the model does not match. " +
                    $"Received TensorProxy of shape [?x{widthBp}x{heightBp}x{pixelBp}] but " +
                    $"was expecting [?x{widthT}x{heightT}x{pixelT}] for the {sensor} Sensor."
                );
            }
            return null;
        }

        /// <summary>
        /// Checks that the shape of the rank 2 observation input placeholder is the same as the corresponding sensor.
        /// </summary>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <param name="sensor">The sensor that produces the visual observation.</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckRankTwoObsShape(
            TensorProxy tensorProxy, ObservationSpec sensor)
        {
            var shape = sensor.Shape;
            var dim1Bp = shape[0];
            var dim2Bp = shape[1];
            var dim1T = tensorProxy.Channels;
            var dim2T = tensorProxy.Width;
            var dim3T = tensorProxy.Height;
            if ((dim1Bp != dim1T) || (dim2Bp != dim2T))
            {
                var proxyDimStr = $"[?x{dim1T}x{dim2T}]";
                if (dim3T > 1)
                {
                    proxyDimStr = $"[?x{dim3T}x{dim2T}x{dim1T}]";
                }
                return FailedCheck.Warning($"An Observation of the model does not match. " +
                    $"Received TensorProxy of shape [?x{dim1Bp}x{dim2Bp}] but " +
                    $"was expecting {proxyDimStr} for the {sensor} Sensor."
                );
            }
            return null;
        }

        /// <summary>
        /// Checks that the shape of the rank 2 observation input placeholder is the same as the corresponding sensor.
        /// </summary>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <param name="sensor">The sensor that produces the visual observation.</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckRankOneObsShape(
            TensorProxy tensorProxy, ObservationSpec sensor)
        {
            var shape = sensor.Shape;
            var dim1Bp = shape[0];
            var dim1T = tensorProxy.Channels;
            var dim2T = tensorProxy.Width;
            var dim3T = tensorProxy.Height;
            if ((dim1Bp != dim1T))
            {
                var proxyDimStr = $"[?x{dim1T}]";
                if (dim2T > 1)
                {
                    proxyDimStr = $"[?x{dim1T}x{dim2T}]";
                }
                if (dim3T > 1)
                {
                    proxyDimStr = $"[?x{dim3T}x{dim2T}x{dim1T}]";
                }
                return FailedCheck.Warning($"An Observation of the model does not match. " +
                    $"Received TensorProxy of shape [?x{dim1Bp}] but " +
                    $"was expecting {proxyDimStr} for the {sensor} Sensor."
                );
            }
            return null;
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="sensors">Attached sensors</param>
        /// <returns>A IEnumerable of the checks that failed</returns>
        static IEnumerable<FailedCheck> CheckInputTensorShape(Model model, ActionSpec actionSpec, ObservationSpec[] sensors)
        {
            var failedModelChecks = new List<FailedCheck>();
            
            var memoryInputs = new HashSet<string>();
                
            foreach (var mem in model.memories)
            {
                memoryInputs.Add(mem.input);
            }

            // Dictionary for storing the kind of check we will execute on this tensor.
            // null indicates no check needed (placeholder tensor)
            var sensorInputs = new Dictionary<string, ObservationSpec>();

            for (var sensorIndex = 0; sensorIndex < sensors.Length; sensorIndex++)
            {
                var observationName = TensorNames.GetObservationName(sensorIndex);
                sensorInputs[observationName] = sensors[sensorIndex];

            }

            // If the model expects an input but it is not in this list
            foreach (var tensor in model.GetInputTensors())
            {
                var error = tensor.name switch
                {
                    TensorNames.PreviousActionPlaceholder => CheckPreviousActionShape(actionSpec, tensor),
                    TensorNames.RandomNormalEpsilonPlaceholder => null,
                    TensorNames.ActionMaskPlaceholder => null,
                    TensorNames.SequenceLengthPlaceholder => null,
                    TensorNames.RecurrentInPlaceholder => null,
                    _ when memoryInputs.Contains(tensor.name) => null,
                    _ when sensorInputs.TryGetValue(tensor.name, out var sensor) => sensor.Rank switch
                    {
                        1 => CheckRankOneObsShape(tensor,sensor),
                        2 => CheckRankTwoObsShape(tensor,sensor),
                        3 => CheckVisualObsShape(tensor,sensor),
                        _ => FailedCheck.Warning("unsupported sensor Rank in inputs: " + sensor.Rank),
                    },
                    _ => FailedCheck.Warning("Model contains an unexpected input named : " + tensor.name),
                };
                
                if (error != null)
                {
                    failedModelChecks.Add(error);
                }
            }
            return failedModelChecks;
        }

        /// <summary>
        /// Checks that the shape of the Previous Vector Action input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="tensorProxy"> The tensor that is expected by the model</param>
        /// <param name="sensors">Array of attached sensor components (unused).</param>
        /// <param name="observableAttributeTotalSize">Sum of the sizes of all ObservableAttributes (unused).</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        static FailedCheck CheckPreviousActionShape(ActionSpec actionSpec, TensorProxy tensorProxy)
        {
            var numberActionsBp = actionSpec.NumDiscreteActions;
            var numberActionsT = tensorProxy.shape[^1];
            if (numberActionsBp != numberActionsT)
            {
                return FailedCheck.Warning("Previous Action Size of the model does not match. " +
                    $"Received {numberActionsBp} but was expecting {numberActionsT}."
                );
            }
            return null;
        }

        /// <summary>
        /// Generates failed checks that correspond to output shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="actuatorComponents">Array of attached actuator components.</param>
        /// <returns>
        /// A IEnumerable of error messages corresponding to the incompatible shapes between model
        /// and BrainParameters.
        /// </returns>
        static IEnumerable<FailedCheck> CheckOutputTensorShape(
            Model model,
            ActionSpec actionSpec)
        {
            var failedModelChecks = new List<FailedCheck>();

            // If the model expects an output but it is not in this list
            var modelContinuousActionSize = model.ContinuousOutputSize();
            var continuousError = CheckContinuousActionOutputShape(actionSpec, modelContinuousActionSize);
            if (continuousError != null)
            {
                failedModelChecks.Add(continuousError);
            }
            FailedCheck discreteError = null;
            var modelApiVersion = model.GetVersion();
            if (modelApiVersion == (int)ModelApiVersion.MLAgents2_0)
            {
                var modelDiscreteBranches = model.GetTensorByName(TensorNames.DiscreteActionOutputShape);
                discreteError = CheckDiscreteActionOutputShape(actionSpec, modelDiscreteBranches);
            }
            else
            {
                throw new NotSupportedException("Version not supported");
            }

            if (discreteError != null)
            {
                failedModelChecks.Add(discreteError);
            }
            return failedModelChecks;
        }

        /// <summary>
        /// Checks that the shape of the discrete action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="actionSpec">
        /// The Actions that the model can take. If multiple actuators are included, merge the ActionSpecs before submitting</param>
        /// <param name="modelDiscreteBranches"> The Tensor of branch sizes.
        /// </param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckDiscreteActionOutputShape(ActionSpec actionSpec, Tensor modelDiscreteBranches)
        {

            var discreteActionBranches = actionSpec.BranchSizes;

            int modelDiscreteBranchesLength = modelDiscreteBranches?.length ?? 0;
            if (modelDiscreteBranchesLength != discreteActionBranches.Length)
            {
                return FailedCheck.Warning("Discrete Action Size of the model does not match. The BrainParameters expect " +
                    $"{discreteActionBranches.Length} branches but the model contains {modelDiscreteBranchesLength}."
                );
            }

            for (int i = 0; i < modelDiscreteBranchesLength; i++)
            {
                if (modelDiscreteBranches != null && modelDiscreteBranches[i] != discreteActionBranches[i])
                {
                    return FailedCheck.Warning($"The number of Discrete Actions of branch {i} does not match. " +
                    $"Was expecting {discreteActionBranches[i]} but the model contains {modelDiscreteBranches[i]} "
                    );
                }
            }
            return null;
        }


        /// <summary>
        /// Checks that the shape of the continuous action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="actuatorComponents">Array of attached actuator components.</param>
        /// <param name="modelContinuousActionSize">
        /// The size of the continuous action output that is expected by the model.
        /// </param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        static FailedCheck CheckContinuousActionOutputShape(ActionSpec actionSpec, int modelContinuousActionSize)
        {
            var numContinuousActions = actionSpec.NumContinuousActions;
            
            if (modelContinuousActionSize != numContinuousActions)
            {
                return FailedCheck.Warning(
                    "Continuous Action Size of the model does not match. The BrainParameters and ActuatorComponents expect " +
                    $"{numContinuousActions} but the model contains {modelContinuousActionSize}."
                );
            }
            return null;
        }
    }
}
