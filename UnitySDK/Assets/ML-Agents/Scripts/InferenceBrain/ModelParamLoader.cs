#if ENABLE_TENSORFLOW
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Prepares the Tensors for the Learning Brain and exposes a list of failed checks if Model
    /// and BrainParameters are incompatible.
    /// </summary>
    public class ModelParamLoader
    {
        private enum ModelActionType
        {
            Unknown,
            Discrete,
            Continuous
        }
        private const long ApiVersion = 2;
        private TFSharpInferenceEngine _engine;
        private BrainParameters _brainParameters;
        private List<string> _failedModelChecks = new List<string>();

        /// <summary>
        /// Factory for the ModelParamLoader : Creates a ModelParamLoader and runs the checks
        /// on it.
        /// </summary>
        /// <param name="engine"> The InferenceEngine we get the parameters and the checks from
        /// </param>
        /// <param name="brainParameters"> The BrainParamters that are used verify the
        /// compatibility with the InferenceEngine</param>
        /// <returns></returns>
        public static ModelParamLoader GetLoaderAndCheck(TFSharpInferenceEngine engine,
            BrainParameters brainParameters)
        {
            ModelParamLoader modelParamLoader = new ModelParamLoader(engine, brainParameters);
            modelParamLoader.GenerateChecks();
            return modelParamLoader;
        }
        
        private ModelParamLoader(TFSharpInferenceEngine engine, BrainParameters brainParameters)
        {
            _engine = engine;
            _brainParameters = brainParameters;
        }

        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model. 
        /// </summary>
        /// <returns>Tensor IEnumerable with the expected Tensor inputs</returns>
        public IReadOnlyList<Tensor> GetInputTensors()
        {
            return _engine?.InputFeatures();
        }

        /// <summary>
        /// Generates the Tensor outputs that are expected to be present in the Model. 
        /// </summary>
        /// <returns>Tensor IEnumerable with the expected Tensor outputs</returns>
        public IReadOnlyList<Tensor> GetOutputTensors()
        {
            var tensorList = new List<Tensor>();
            if (_brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.ActionOutput,
                    Shape = new long[]
                    {
                        -1, _brainParameters.vectorActionSize[0]
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null
                });
            }
            else
            {
                tensorList.Add(
                    new Tensor()
                    {
                        Name = TensorNames.ActionOutput,
                        Shape = new long[]
                        {
                            -1, _brainParameters.vectorActionSize.Sum()
                        },
                        ValueType = Tensor.TensorType.FloatingPoint,
                        Data = null
                    });
            }
            var memory = GetIntScalar(TensorNames.MemorySize);
            if (memory > 0)
            {
                tensorList.Add(new Tensor()
                {
                    Name = TensorNames.RecurrentOutput,
                    Shape = new long[2]
                    {
                        -1, memory
                    },
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null
                });
            }
            return tensorList;
        }

        /// <summary>
        /// Queries the InferenceEngine for the value of a variable in the graph given its name.
        /// Only works with int32 Tensors with zero dimensions containing a unique element.
        /// If the node was not found or could not be retrieved, the value -1 will be returned. 
        /// </summary>
        /// <param name="name">The name of the Tensor variable</param>
        /// <returns>The value of the scalar variable in the model. (-1 if not found)</returns>
        private int GetIntScalar(string name)
        {
            var outputs = new Tensor[]
            {
                new Tensor()
                {
                    Name = name,
                    ValueType = Tensor.TensorType.Integer,
                    Shape = new long[] { },
                    Data = new long[1]
                },
            };
            try
            {
                _engine.ExecuteGraph(new Tensor[0], outputs);
            }
            catch
            {
                return -1;
            }
            return (outputs[0].Data as int[])[0];
        }

        /// <summary>
        /// Retrieves an IEnumerable of string corresponding to the failed compatibility checks
        /// between the InferenceEngine and the BrainParameters. 
        /// </summary>
        public IEnumerable<string> GetChecks()
        {
            return _failedModelChecks;
        }

        /// <summary>
        /// Generates the list of failed checks that failed when comparing the data from the Model
        /// and from the BrainParameters
        /// </summary>
        private void GenerateChecks()
        {
            _failedModelChecks.Clear();
            if (_engine == null)
            {
                _failedModelChecks.Add(
                    "There is no model for this Brain, cannot run inference. " +
                    "(But can still train)");
                return;
            }

            var modelApiVersion = GetIntScalar(TensorNames.VersionNumber);
            var memorySize = GetIntScalar(TensorNames.MemorySize);
            var isContinuousInt = GetIntScalar(TensorNames.IsContinuousControl);
            var isContinuous = GetActionType(isContinuousInt);
            var actionSize = GetIntScalar(TensorNames.ActionOutputShape);
            if (modelApiVersion == -1)
            {
                _failedModelChecks.Add(
                    "Model was not trained using the right version of ML-Agents. Cannot use this " +
                    "model.");
                return;
            }
            if (modelApiVersion != ApiVersion)
            {
                _failedModelChecks.Add(
                    $"Version of the trainer the model was trained with ({modelApiVersion}) " +
                    $"is not compatible with the Brain's version ({ApiVersion}).");
                return;
            }

            CheckIntScalarPresenceHelper(new Dictionary<string, int>()
            {
                {TensorNames.MemorySize, memorySize},
                {TensorNames.IsContinuousControl, isContinuousInt},
                {TensorNames.ActionOutputShape, actionSize}
            });
            CheckInputTensorPresence(memorySize, isContinuous);
            CheckOutputTensorPresence(memorySize);
            CheckInputTensorShape();
            CheckOutputTensorShape(isContinuous, actionSize);
        }

        /// <summary>
        /// Converts the integer value in the model corresponding to the type of control to a
        /// ModelActionType.
        /// </summary>
        /// <param name="isContinuousInt"> The integer value in the model indicating the
        /// type of control</param>
        /// <returns>The equivalent ModelActionType</returns>
        private static ModelActionType GetActionType(int isContinuousInt)
        {
            ModelActionType isContinuous;
            switch (isContinuousInt)
            {
                case 0:
                    isContinuous = ModelActionType.Discrete;
                    break;
                case 1: 
                    isContinuous = ModelActionType.Continuous;
                    break;
                default:
                    isContinuous = ModelActionType.Unknown;
                    break;
            }
            return isContinuous;
        }

        /// <summary>
        /// Given a Dictionary of node names to int values, create checks if the values have the
        /// invalid value of -1.
        /// </summary>
        /// <param name="requiredScalarFields"> Mapping from node names to int values</param>
        private void CheckIntScalarPresenceHelper(Dictionary<string, int> requiredScalarFields)
        {
            foreach(var field in requiredScalarFields)
            if (field.Value == -1)
            {
                _failedModelChecks.Add(
                    $"Missing node in the model provided : {field.Key}");
            }
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="memory"> The memory size that the model is expecting/</param>
        /// <param name="isContinuous"> Whether the model is expecting continuous or
        /// discrete control.</param>
        /// <returns>A IEnumerable of string corresponding to the failed input presence
        /// checks.</returns>
        private void CheckInputTensorPresence(int memory, ModelActionType isContinuous)
        {
            var tensorsNames = GetInputTensors().Select(x => x.Name).ToList();
            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((_brainParameters.vectorObservationSize != 0) &&
                (!tensorsNames.Contains(TensorNames.VectorObservationPlacholder)))
            {
                _failedModelChecks.Add(
                    "The model does not contain a Vector Observation  Placeholder Input. " +
                    "You must set the Vector Observation Space Size to 0.");
            }
            // If there are not enough Visual Observation Input compared to what the
            // Brain Parameters expect.
            for (var visObsIndex = 0;
                visObsIndex < _brainParameters.cameraResolutions.Length;
                visObsIndex++)
            {
                if (!tensorsNames.Contains(
                    TensorNames.VisualObservationPlaceholderPrefix + visObsIndex))
                {
                    _failedModelChecks.Add(
                        "The model does not contain a Visual Observation Placeholder Input " +
                        "for visual observation "+visObsIndex+".");
                }
            }
            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                if (!tensorsNames.Contains(TensorNames.RecurrentInPlaceholder))
                {
                    _failedModelChecks.Add(
                        "The model does not contain a Recurrent Input Node but has memory_size.");
                }
            }
            // If the model uses discrete control but does not have an input for action masks
            if (isContinuous == ModelActionType.Discrete)
            {
                if (!tensorsNames.Contains(TensorNames.ActionMaskPlaceholder))
                {
                    _failedModelChecks.Add(
                        "The model does not contain an Action Mask but is using Discrete Control.");
                }
            }
        }
        
        /// <summary>
        /// Generates failed checks that correspond to outputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="memory"> The memory size that the model is expecting/</param>
        /// <returns>A IEnumerable of string corresponding to the failed output presence
        /// checks.</returns>
        private void CheckOutputTensorPresence(int memory)
        {
            var tensorsNames = GetOutputTensors().Select(x => x.Name).ToList();
            // If there is no Action Output.
            if (!tensorsNames.Contains(TensorNames.ActionOutput))
            {
                _failedModelChecks.Add("The model does not contain an Action Output Node.");
            }
            
            // If there is no Recurrent Output but the model is Recurrent.
            if (memory > 0)
            {
                if (!tensorsNames.Contains(TensorNames.RecurrentOutput))
                {
                    _failedModelChecks.Add(
                        "The model does not contain a Recurrent Output Node but has memory_size.");
                }
            }
        }
        
        /// <summary>
        /// Generates failed checks that correspond to inputs shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        private void CheckInputTensorShape()
        {
            var tensorTester =
                new Dictionary<string, Func<Tensor, string>>()
                {
                    {TensorNames.VectorObservationPlacholder, CheckVectorObsShape},
                    {TensorNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                    {TensorNames.RandomNormalEpsilonPlaceholder, ((tensor) => null)},
                    {TensorNames.ActionMaskPlaceholder, ((tensor) => null)},
                    {TensorNames.SequenceLengthPlaceholder, ((tensor) => null)},
                    {TensorNames.RecurrentInPlaceholder, ((tensor) => null)},
                };
            for (var obsIndex = 0; obsIndex < _brainParameters.cameraResolutions.Length; obsIndex++)
            {
                var index = obsIndex;
                tensorTester[TensorNames.VisualObservationPlaceholderPrefix + obsIndex] =
                    (tensor) => CheckVisualObsShape(tensor, index);
            }
            // If the model expects an input but it is not in this list
            foreach (var tensor in GetInputTensors())
            {
                if (!tensorTester.ContainsKey(tensor.Name))
                {
                    _failedModelChecks.Add(
                        "Model requires an unknown input named : " + tensor.Name);
                }
                else
                {
                    var tester = tensorTester[tensor.Name];
                    var error = tester.Invoke(tensor);
                    if (error != null)
                    {
                        _failedModelChecks.Add(error);
                    }
                }
            }
        }
        
        /// <summary>
        /// Checks that the shape of the Vector Observation input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensor"> The tensor that is expected by the model</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckVectorObsShape(Tensor tensor)
        {
            var vecObsSizeBp = _brainParameters.vectorObservationSize;
            var numStackedVector = _brainParameters.numStackedVectorObservations;
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
        
        /// <summary>
        /// Checks that the shape of the Previous Vector Action input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensor"> The tensor that is expected by the model</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckPreviousActionShape(Tensor tensor)
        {
            var numberActionsBp = _brainParameters.vectorActionSize.Length;
            var numberActionsT = tensor.Shape[1];
            if  (numberActionsBp != numberActionsT)
            {
                return string.Format(
                    "Previous Action Size of the model does not match. " +
                    "Received {0} but was expecting {1}.",
                    numberActionsBp, numberActionsT);
            }
            return null;
        }
        
        /// <summary>
        /// Checks that the shape of the visual observation input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensor"> The tensor that is expected by the model</param>
        /// <param name="visObsIndex"> The index of the visual observation.</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckVisualObsShape(Tensor tensor, int visObsIndex)
        {
            var resolutionBp = _brainParameters.cameraResolutions[visObsIndex];
            var widthBp = resolutionBp.width;
            var heightBp = resolutionBp.height;
            var pixelBp = resolutionBp.blackAndWhite ? 1 : 3;
            var heightT = tensor.Shape[1];
            var widthT = tensor.Shape[2];
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
        
        /// <summary>
        /// Generates failed checks that correspond to output shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        /// <param name="isContinuous"> Whether the model is expecting continuous or
        /// discrete control.</param>
        /// <param name="modelActionSize"> The size of the action output that is expected
        /// by the model.</param>
        /// <returns>A IEnumerable of string corresponding to the incompatible shapes between
        /// model and BrainParameters.</returns>
        private void CheckOutputTensorShape(ModelActionType isContinuous, int modelActionSize)
        {
            if (isContinuous == ModelActionType.Unknown)
            {
                _failedModelChecks.Add(
                    "Cannot infer type of Control from the provided model.");
                return;
            }
            if (isContinuous == ModelActionType.Continuous &&
                _brainParameters.vectorActionSpaceType != SpaceType.continuous)
            {
                _failedModelChecks.Add(
                    "Model has been trained using Continuous Control but the Brain Parameters " +
                    "suggest Discrete Control.");
                return;
            }
            if (isContinuous == ModelActionType.Discrete &&
                _brainParameters.vectorActionSpaceType != SpaceType.discrete)
            {
                _failedModelChecks.Add(
                    "Model has been trained using Discrete Control but the Brain Parameters " +
                    "suggest Continuous Control.");
                return;
            }
            var tensorTester = new Dictionary<string, Func<Tensor, int, string>>();
            if (_brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorTester[TensorNames.ActionOutput] = CheckContinuousActionOutputShape;
            }
            else
            {
                tensorTester[TensorNames.ActionOutput] = CheckDiscreteActionOutputShape;
            }
            // If the model expects an output but it is not in this list
            foreach (var tensor in GetOutputTensors())
            {
                if (tensorTester.ContainsKey(tensor.Name))
                {
                    var tester = tensorTester[tensor.Name];
                    var error = tester.Invoke(tensor, modelActionSize);
                    if (error != null)
                    {
                        _failedModelChecks.Add(error);
                    }
                }
            }
        }

        /// <summary>
        /// Checks that the shape of the discrete action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensor"> The tensor that is expected by the model</param>
        /// <param name="modelActionSize"> The size of the action output that is expected
        /// by the model.</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckDiscreteActionOutputShape(Tensor tensor, int modelActionSize)
        {
            var bpActionSize = _brainParameters.vectorActionSize.Sum();
            if  (modelActionSize != bpActionSize)
            {
                return string.Format(
                    "Action Size of the model does not match. " +
                    "The BrainParameters expect {0} but the model contains {1}.",
                    bpActionSize, modelActionSize);
            }
            return null;
        }
        
        /// <summary>
        /// Checks that the shape of the continuous action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensor"> The tensor that is expected by the model</param>
        /// <param name="modelActionSize"> The size of the action output that is expected
        /// by the model.</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckContinuousActionOutputShape(Tensor tensor, int modelActionSize)
        {
            var bpActionSize = _brainParameters.vectorActionSize[0];
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
#endif
