using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Prepares the Tensors for the Learning Brain and exposes a list of failed checks if Model
    /// and BrainParameters are incompatible.
    /// </summary>
    internal class SentisModelParamLoader
    {
        internal enum ModelApiVersion
        {
            /// <summary>
            /// ML-Agents model version for versions 1.x.y
            /// The observations are split between vector and visual observations
            /// There are legacy action outputs for discrete and continuous actions
            /// LSTM inputs and outputs are handled by Sentis
            /// </summary>
            MLAgents1_0 = 2,

            /// <summary>
            /// All observations are treated the same and named obs_{i} with i being
            /// the sensor index
            /// Legacy "action" output is no longer present
            /// LSTM inputs and outputs are treated like regular inputs and outputs
            /// and no longer managed by Sentis
            /// </summary>
            MLAgents2_0 = 3,
            MinSupportedVersion = MLAgents1_0,
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
        /// The Sentis engine model for loading static parameters
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

            var memorySize = (int)((TensorFloat)model.GetTensorByName(TensorNames.MemorySize))[0];

            if (modelApiVersion == (int)ModelApiVersion.MLAgents1_0 && memorySize > 0)
            {
                // This block is to make sure that models that are trained with MLAgents version 1.x and have
                // an LSTM (i.e. use the Sentis _c and _h inputs and outputs) will not work with MLAgents version
                // 2.x. This is because Sentis version 2.x will eventually drop support for the _c and _h inputs
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
        /// The Sentis engine model for loading static parameters
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
            BrainParameters brainParameters,
            ISensor[] sensors,
            ActuatorComponent[] actuatorComponents,
            int observableAttributeTotalSize = 0,
            BehaviorType behaviorType = BehaviorType.Default,
            bool deterministicInference = false
        )
        {
            List<FailedCheck> failedModelChecks = new List<FailedCheck>();
            if (model == null)
            {
                var errorMsg = "There is no model for this Brain; cannot run inference. ";
                if (behaviorType == BehaviorType.InferenceOnly)
                {
                    errorMsg += "Either assign a model, or change to a different Behavior Type.";
                }
                else
                {
                    errorMsg += "(But can still train)";
                }
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

            var memorySize = (int)((TensorFloat)model.GetTensorByName(TensorNames.MemorySize))[0];
            if (memorySize == -1)
            {
                failedModelChecks.Add(FailedCheck.Warning($"Missing node in the model provided : {TensorNames.MemorySize}"
                ));
                return failedModelChecks;
            }

            if (modelApiVersion == (int)ModelApiVersion.MLAgents1_0)
            {
                failedModelChecks.AddRange(
                    CheckInputTensorPresenceLegacy(model, brainParameters, memorySize, sensors)
                );
                failedModelChecks.AddRange(
                    CheckInputTensorShapeLegacy(model, brainParameters, sensors, observableAttributeTotalSize)
                );
            }
            else if (modelApiVersion == (int)ModelApiVersion.MLAgents2_0)
            {
                failedModelChecks.AddRange(
                    CheckInputTensorPresence(model, brainParameters, memorySize, sensors, deterministicInference)
                );
                failedModelChecks.AddRange(
                    CheckInputTensorShape(model, brainParameters, sensors, observableAttributeTotalSize)
                );
            }


            failedModelChecks.AddRange(
                CheckOutputTensorShape(model, brainParameters, actuatorComponents)
            );

            failedModelChecks.AddRange(
                CheckOutputTensorPresence(model, memorySize, deterministicInference)
            );
            return failedModelChecks;
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs expected by the model that are not
        /// present in the BrainParameters. Tests the models created with the API of version 1.X
        /// </summary>
        /// <param name="model">
        /// The Sentis engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="memory">
        /// The memory size that the model is expecting.
        /// </param>
        /// <param name="sensors">Array of attached sensor components</param>
        /// <returns>
        /// A IEnumerable of the checks that failed
        /// </returns>
        static IEnumerable<FailedCheck> CheckInputTensorPresenceLegacy(
            Model model,
            BrainParameters brainParameters,
            int memory,
            ISensor[] sensors
        )
        {
            var failedModelChecks = new List<FailedCheck>();
            var tensorsNames = model.GetInputNames();

            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((brainParameters.VectorObservationSize != 0) &&
                (!tensorsNames.Contains(TensorNames.VectorObservationPlaceholder)))
            {
                failedModelChecks.Add(
                    FailedCheck.Warning("The model does not contain a Vector Observation Placeholder Input. " +
                        "You must set the Vector Observation Space Size to 0.")
                );
            }

            // If there are not enough Visual Observation Input compared to what the
            // sensors expect.
            var visObsIndex = 0;
            for (var sensorIndex = 0; sensorIndex < sensors.Length; sensorIndex++)
            {
                var sensor = sensors[sensorIndex];
                if (sensor.GetObservationSpec().Shape.Length == 3)
                {
                    if (!tensorsNames.Contains(
                        TensorNames.GetVisualObservationName(visObsIndex)))
                    {
                        failedModelChecks.Add(
                            FailedCheck.Warning("The model does not contain a Visual Observation Placeholder Input " +
                                $"for sensor component {visObsIndex} ({sensor.GetType().Name}).")
                        );
                    }
                    visObsIndex++;
                }
                if (sensor.GetObservationSpec().Shape.Length == 2)
                {
                    if (!tensorsNames.Contains(
                        TensorNames.GetObservationName(sensorIndex)))
                    {
                        failedModelChecks.Add(
                            FailedCheck.Warning("The model does not contain an Observation Placeholder Input " +
                                $"for sensor component {sensorIndex} ({sensor.GetType().Name}).")
                        );
                    }
                }
            }

            var expectedVisualObs = model.GetNumVisualInputs();
            // Check if there's not enough visual sensors (too many would be handled above)
            if (expectedVisualObs > visObsIndex)
            {
                failedModelChecks.Add(
                    FailedCheck.Warning($"The model expects {expectedVisualObs} visual inputs," +
                        $" but only found {visObsIndex} visual sensors.")
                );
            }

            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                if (!tensorsNames.Any(x => x.EndsWith("_h")) ||
                    !tensorsNames.Any(x => x.EndsWith("_c")))
                {
                    failedModelChecks.Add(
                        FailedCheck.Warning("The model does not contain a Recurrent Input Node but has memory_size.")
                    );
                }
            }

            // If the model uses discrete control but does not have an input for action masks
            if (model.HasDiscreteOutputs())
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
        /// Generates failed checks that correspond to inputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Sentis engine model for loading static parameters
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
            BrainParameters brainParameters,
            int memory,
            ISensor[] sensors,
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
                            $"for sensor component {sensorIndex} ({sensor.GetType().Name}).")
                    );
                }
            }

            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                var modelVersion = model.GetVersion();
                if (!tensorsNames.Any(x => x == TensorNames.RecurrentInPlaceholder))
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
        /// The Sentis engine model for loading static parameters
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
                if (!allOutputs.Any(x => x == TensorNames.RecurrentOutput))
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
            TensorProxy tensorProxy, ISensor sensor)
        {
            var shape = sensor.GetObservationSpec().Shape;
            var heightBp = shape[1];
            var widthBp = shape[2];
            var pixelBp = shape[0];
            var heightT = tensorProxy.Height;
            var widthT = tensorProxy.Width;
            var pixelT = tensorProxy.Channels;
            if ((widthBp != widthT) || (heightBp != heightT) || (pixelBp != pixelT))
            {
                return FailedCheck.Warning($"The visual Observation of the model does not match. " +
                    $"Received TensorProxy of shape [?x{widthBp}x{heightBp}x{pixelBp}] but " +
                    $"was expecting [?x{widthT}x{heightT}x{pixelT}] for the {sensor.GetName()} Sensor."
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
            TensorProxy tensorProxy, ISensor sensor)
        {
            var shape = sensor.GetObservationSpec().Shape;
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
                    $"was expecting {proxyDimStr} for the {sensor.GetName()} Sensor."
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
            TensorProxy tensorProxy, ISensor sensor)
        {
            var shape = sensor.GetObservationSpec().Shape;
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
                    $"was expecting {proxyDimStr} for the {sensor.GetName()} Sensor."
                );
            }
            return null;
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs shapes incompatibilities between
        /// the model and the BrainParameters. Tests the models created with the API of version 1.X
        /// </summary>
        /// <param name="model">
        /// The Sentis engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="sensors">Attached sensors</param>
        /// <param name="observableAttributeTotalSize">Sum of the sizes of all ObservableAttributes.</param>
        /// <returns>A IEnumerable of the checks that failed</returns>
        static IEnumerable<FailedCheck> CheckInputTensorShapeLegacy(
            Model model, BrainParameters brainParameters, ISensor[] sensors,
            int observableAttributeTotalSize)
        {
            var failedModelChecks = new List<FailedCheck>();
            var tensorTester =
                new Dictionary<string, Func<BrainParameters, TensorProxy, ISensor[], int, FailedCheck>>()
            {
                {TensorNames.VectorObservationPlaceholder, CheckVectorObsShapeLegacy},
                {TensorNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                {TensorNames.RandomNormalEpsilonPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.ActionMaskPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.SequenceLengthPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.RecurrentInPlaceholder, ((bp, tensor, scs, i) => null)},
            };

            // foreach (var mem in model.memories)
            // {
            //     tensorTester[mem.input] = ((bp, tensor, scs, i) => null);
            // }

            var visObsIndex = 0;
            for (var sensorIndex = 0; sensorIndex < sensors.Length; sensorIndex++)
            {
                var sens = sensors[sensorIndex];
                if (sens.GetObservationSpec().Shape.Length == 3)
                {
                    tensorTester[TensorNames.GetVisualObservationName(visObsIndex)] =
                        (bp, tensor, scs, i) => CheckVisualObsShape(tensor, sens);
                    visObsIndex++;
                }
                if (sens.GetObservationSpec().Shape.Length == 2)
                {
                    tensorTester[TensorNames.GetObservationName(sensorIndex)] =
                        (bp, tensor, scs, i) => CheckRankTwoObsShape(tensor, sens);
                }
            }

            // If the model expects an input but it is not in this list
            foreach (var tensor in model.GetInputTensors())
            {
                if (!tensorTester.ContainsKey(tensor.name))
                {
                    if (!tensor.name.Contains("visual_observation"))
                    {
                        failedModelChecks.Add(
                            FailedCheck.Warning("Model contains an unexpected input named : " + tensor.name)
                        );
                    }
                }
                else
                {
                    var tester = tensorTester[tensor.name];
                    var error = tester.Invoke(brainParameters, tensor, sensors, observableAttributeTotalSize);
                    if (error != null)
                    {
                        failedModelChecks.Add(error);
                    }
                }
            }
            return failedModelChecks;
        }

        /// <summary>
        /// Checks that the shape of the Vector Observation input placeholder is the same in the
        /// model and in the Brain Parameters. Tests the models created with the API of version 1.X
        /// </summary>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <param name="sensors">Array of attached sensor components</param>
        /// <param name="observableAttributeTotalSize">Sum of the sizes of all ObservableAttributes.</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckVectorObsShapeLegacy(
            BrainParameters brainParameters, TensorProxy tensorProxy, ISensor[] sensors,
            int observableAttributeTotalSize)
        {
            var vecObsSizeBp = brainParameters.VectorObservationSize;
            var numStackedVector = brainParameters.NumStackedVectorObservations;
            var totalVecObsSizeT = tensorProxy.shape[tensorProxy.shape.Length - 1];

            var totalVectorSensorSize = 0;
            foreach (var sens in sensors)
            {
                if ((sens.GetObservationSpec().Shape.Length == 1))
                {
                    totalVectorSensorSize += sens.GetObservationSpec().Shape[0];
                }
            }

            if (totalVectorSensorSize != totalVecObsSizeT)
            {
                var sensorSizes = "";
                foreach (var sensorComp in sensors)
                {
                    if (sensorComp.GetObservationSpec().Shape.Length == 1)
                    {
                        var vecSize = sensorComp.GetObservationSpec().Shape[0];
                        if (sensorSizes.Length == 0)
                        {
                            sensorSizes = $"[{vecSize}";
                        }
                        else
                        {
                            sensorSizes += $", {vecSize}";
                        }
                    }
                }

                sensorSizes += "]";
                return FailedCheck.Warning(
                    $"Vector Observation Size of the model does not match. Was expecting {totalVecObsSizeT} " +
                    $"but received: \n" +
                    $"Vector observations: {vecObsSizeBp} x {numStackedVector}\n" +
                    $"Total [Observable] attributes: {observableAttributeTotalSize}\n" +
                    $"Sensor sizes: {sensorSizes}."
                );
            }
            return null;
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        /// <param name="model">
        /// The Sentis engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="sensors">Attached sensors</param>
        /// <param name="observableAttributeTotalSize">Sum of the sizes of all ObservableAttributes.</param>
        /// <returns>A IEnumerable of the checks that failed</returns>
        static IEnumerable<FailedCheck> CheckInputTensorShape(
            Model model, BrainParameters brainParameters, ISensor[] sensors,
            int observableAttributeTotalSize)
        {
            var failedModelChecks = new List<FailedCheck>();
            var tensorTester =
                new Dictionary<string, Func<BrainParameters, TensorProxy, ISensor[], int, FailedCheck>>()
            {
                {TensorNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                {TensorNames.RandomNormalEpsilonPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.ActionMaskPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.SequenceLengthPlaceholder, ((bp, tensor, scs, i) => null)},
                {TensorNames.RecurrentInPlaceholder, ((bp, tensor, scs, i) => null)},
            };

            // foreach (var mem in model.memories)
            // {
            //     tensorTester[mem.input] = ((bp, tensor, scs, i) => null);
            // }

            for (var sensorIndex = 0; sensorIndex < sensors.Length; sensorIndex++)
            {
                var sens = sensors[sensorIndex];
                if (sens.GetObservationSpec().Rank == 3)
                {
                    tensorTester[TensorNames.GetObservationName(sensorIndex)] =
                        (bp, tensor, scs, i) => CheckVisualObsShape(tensor, sens);
                }
                if (sens.GetObservationSpec().Rank == 2)
                {
                    tensorTester[TensorNames.GetObservationName(sensorIndex)] =
                        (bp, tensor, scs, i) => CheckRankTwoObsShape(tensor, sens);
                }
                if (sens.GetObservationSpec().Rank == 1)
                {
                    tensorTester[TensorNames.GetObservationName(sensorIndex)] =
                        (bp, tensor, scs, i) => CheckRankOneObsShape(tensor, sens);
                }
            }

            // If the model expects an input but it is not in this list
            foreach (var tensor in model.GetInputTensors())
            {
                if (!tensorTester.ContainsKey(tensor.name))
                {
                    failedModelChecks.Add(FailedCheck.Warning("Model contains an unexpected input named : " + tensor.name
                    ));
                }
                else
                {
                    var tester = tensorTester[tensor.name];
                    var error = tester.Invoke(brainParameters, tensor, sensors, observableAttributeTotalSize);
                    if (error != null)
                    {
                        failedModelChecks.Add(error);
                    }
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
        static FailedCheck CheckPreviousActionShape(
            BrainParameters brainParameters, TensorProxy tensorProxy,
            ISensor[] sensors, int observableAttributeTotalSize)
        {
            var numberActionsBp = brainParameters.ActionSpec.NumDiscreteActions;
            var numberActionsT = tensorProxy.shape[tensorProxy.shape.Length - 1];
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
        /// The Sentis engine model for loading static parameters
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
            BrainParameters brainParameters,
            ActuatorComponent[] actuatorComponents)
        {
            var failedModelChecks = new List<FailedCheck>();

            // If the model expects an output but it is not in this list
            var modelContinuousActionSize = model.ContinuousOutputSize();
            var continuousError = CheckContinuousActionOutputShape(brainParameters, actuatorComponents, modelContinuousActionSize);
            if (continuousError != null)
            {
                failedModelChecks.Add(continuousError);
            }
            FailedCheck discreteError = null;
            var modelApiVersion = model.GetVersion();
            if (modelApiVersion == (int)ModelApiVersion.MLAgents1_0)
            {
                var modelSumDiscreteBranchSizes = model.DiscreteOutputSize();
                discreteError = CheckDiscreteActionOutputShapeLegacy(brainParameters, actuatorComponents, modelSumDiscreteBranchSizes);
            }
            if (modelApiVersion == (int)ModelApiVersion.MLAgents2_0)
            {
                var modelDiscreteBranches = (TensorFloat)model.GetTensorByName(TensorNames.DiscreteActionOutputShape);
                discreteError = CheckDiscreteActionOutputShape(brainParameters, actuatorComponents, modelDiscreteBranches);
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
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="actuatorComponents">Array of attached actuator components.</param>
        /// <param name="modelDiscreteBranches"> The Tensor of branch sizes.
        /// </param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckDiscreteActionOutputShape(
            BrainParameters brainParameters, ActuatorComponent[] actuatorComponents, TensorFloat modelDiscreteBranches)
        {
            var discreteActionBranches = brainParameters.ActionSpec.BranchSizes.ToList();
            foreach (var actuatorComponent in actuatorComponents)
            {
                var actionSpec = actuatorComponent.ActionSpec;
                discreteActionBranches.AddRange(actionSpec.BranchSizes);
            }

            int modelDiscreteBranchesLength = modelDiscreteBranches?.shape.length ?? 0;
            if (modelDiscreteBranchesLength != discreteActionBranches.Count)
            {
                return FailedCheck.Warning("Discrete Action Size of the model does not match. The BrainParameters expect " +
                    $"{discreteActionBranches.Count} branches but the model contains {modelDiscreteBranchesLength}."
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
        /// Checks that the shape of the discrete action output is the same in the
        /// model and in the Brain Parameters. Tests the models created with the API of version 1.X
        /// </summary>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <param name="actuatorComponents">Array of attached actuator components.</param>
        /// <param name="modelSumDiscreteBranchSizes">
        /// The size of the discrete action output that is expected by the model.
        /// </param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        static FailedCheck CheckDiscreteActionOutputShapeLegacy(
            BrainParameters brainParameters, ActuatorComponent[] actuatorComponents, int modelSumDiscreteBranchSizes)
        {
            // TODO: check each branch size instead of sum of branch sizes
            var sumOfDiscreteBranchSizes = brainParameters.ActionSpec.SumOfDiscreteBranchSizes;

            foreach (var actuatorComponent in actuatorComponents)
            {
                var actionSpec = actuatorComponent.ActionSpec;
                sumOfDiscreteBranchSizes += actionSpec.SumOfDiscreteBranchSizes;
            }

            if (modelSumDiscreteBranchSizes != sumOfDiscreteBranchSizes)
            {
                return FailedCheck.Warning("Discrete Action Size of the model does not match. The BrainParameters expect " +
                    $"{sumOfDiscreteBranchSizes} but the model contains {modelSumDiscreteBranchSizes}."
                );
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
        static FailedCheck CheckContinuousActionOutputShape(
            BrainParameters brainParameters, ActuatorComponent[] actuatorComponents, int modelContinuousActionSize)
        {
            var numContinuousActions = brainParameters.ActionSpec.NumContinuousActions;

            foreach (var actuatorComponent in actuatorComponents)
            {
                var actionSpec = actuatorComponent.ActionSpec;
                numContinuousActions += actionSpec.NumContinuousActions;
            }

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
