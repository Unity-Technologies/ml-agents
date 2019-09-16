using System;
using System.Collections.Generic;
using System.Linq;
using Barracuda;

namespace MLAgents.InferenceBrain
{
    /// <summary>
    /// Prepares the Tensors for the Learning Brain and exposes a list of failed checks if Model
    /// and BrainParameters are incompatible.
    /// </summary>
    public class BarracudaModelParamLoader
    {
        private enum ModelActionType
        {
            Unknown,
            Discrete,
            Continuous
        }
        private const long k_ApiVersion = 2;
        private readonly IWorker m_Engine;
        private readonly Model m_Model;
        private readonly BrainParameters m_BrainParameters;
        private readonly List<string> m_FailedModelChecks = new List<string>();

        /// <summary>
        /// Factory for the ModelParamLoader : Creates a ModelParamLoader and runs the checks
        /// on it.
        /// </summary>
        /// <param name="engine">
        /// The Barracuda engine worker we get the parameters and the checks from
        /// </param>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters">
        /// The BrainParameters that are used verify the compatibility with the InferenceEngine
        /// </param>
        /// <returns></returns>
        public static BarracudaModelParamLoader GetLoaderAndCheck(
            IWorker engine, Model model, BrainParameters brainParameters)
        {
            var modelParamLoader = new BarracudaModelParamLoader(engine, model, brainParameters);
            modelParamLoader.GenerateChecks();
            return modelParamLoader;
        }

        private BarracudaModelParamLoader(
            IWorker engine, Model model, BrainParameters brainParameters)
        {
            m_Engine = engine;
            m_Model = model;
            m_BrainParameters = brainParameters;
        }

        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model.
        /// </summary>
        /// <returns>TensorProxy IEnumerable with the expected Tensor inputs</returns>
        public IReadOnlyList<TensorProxy> GetInputTensors()
        {
            var tensors = new List<TensorProxy>();

            if (m_Model == null)
                return tensors;

            foreach (var input in m_Model.inputs)
            {
                tensors.Add(new TensorProxy
                {
                    name = input.name,
                    valueType = TensorProxy.TensorType.FloatingPoint,
                    data = null,
                    shape = input.shape.Select(i => (long)i).ToArray()
                });
            }

            foreach (var mem in m_Model.memories)
            {
                //Debug.Log($"{mem.input}: {mem.shape} -> {BarracudaUtils.TensorShapeFromBarracuda(mem.shape).Length}");
                tensors.Add(new TensorProxy
                {
                    name = mem.input,
                    valueType = TensorProxy.TensorType.FloatingPoint,
                    data = null,
                    shape = TensorUtils.TensorShapeFromBarracuda(mem.shape)
                });
            }

            tensors.Sort((el1, el2) => el1.name.CompareTo(el2.name));

            return tensors;
        }

        /// <summary>
        /// Generates the Tensor outputs that are expected to be present in the Model.
        /// </summary>
        /// <returns>TensorProxy IEnumerable with the expected Tensor outputs</returns>
        public string[] GetOutputNames()
        {
            var names = new List<string>();

            if (m_Model == null)
            {
                return names.ToArray();
            }

            names.Add(TensorNames.ActionOutput);

            var memory = GetIntScalar(TensorNames.MemorySize);
            if (memory > 0)
            {
                foreach (var mem in m_Model.memories)
                {
                    names.Add(mem.output);
                }
            }

            names.Sort();

            return names.ToArray();
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
            return (int)m_Model.GetTensorByName(name)[0];
        }

        /// <summary>
        /// Retrieves an IEnumerable of string corresponding to the failed compatibility checks
        /// between the InferenceEngine and the BrainParameters.
        /// </summary>
        public IEnumerable<string> GetChecks()
        {
            return m_FailedModelChecks;
        }

        /// <summary>
        /// Generates the list of failed checks that failed when comparing the data from the Model
        /// and from the BrainParameters
        /// </summary>
        private void GenerateChecks()
        {
            m_FailedModelChecks.Clear();
            if (m_Engine == null)
            {
                m_FailedModelChecks.Add(
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
                m_FailedModelChecks.Add(
                    "Model was not trained using the right version of ML-Agents. " +
                    "Cannot use this model.");
                return;
            }
            if (modelApiVersion != k_ApiVersion)
            {
                m_FailedModelChecks.Add(
                    $"Version of the trainer the model was trained with ({modelApiVersion}) " +
                    $"is not compatible with the Brain's version ({k_ApiVersion}).");
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
        /// <param name="isContinuousInt">
        /// The integer value in the model indicating the type of control
        /// </param>
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
            foreach (var field in requiredScalarFields)
            {
                if (field.Value == -1)
                {
                    m_FailedModelChecks.Add($"Missing node in the model provided : {field.Key}");
                }
            }
        }

        /// <summary>
        /// Generates failed checks that correspond to inputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="memory">
        /// The memory size that the model is expecting.
        /// </param>
        /// <param name="isContinuous">
        /// Whether the model is expecting continuous or discrete control.
        /// </param>
        /// <returns>
        /// A IEnumerable of string corresponding to the failed input presence checks.
        /// </returns>
        private void CheckInputTensorPresence(int memory, ModelActionType isContinuous)
        {
            var tensorsNames = GetInputTensors().Select(x => x.name).ToList();

            // If there is no Vector Observation Input but the Brain Parameters expect one.
            if ((m_BrainParameters.vectorObservationSize != 0) &&
                (!tensorsNames.Contains(TensorNames.VectorObservationPlacholder)))
            {
                m_FailedModelChecks.Add(
                    "The model does not contain a Vector Observation  Placeholder Input. " +
                    "You must set the Vector Observation Space Size to 0.");
            }
            // If there are not enough Visual Observation Input compared to what the
            // Brain Parameters expect.
            for (var visObsIndex = 0;
                 visObsIndex < m_BrainParameters.cameraResolutions.Length;
                 visObsIndex++)
            {
                if (!tensorsNames.Contains(
                    TensorNames.VisualObservationPlaceholderPrefix + visObsIndex))
                {
                    m_FailedModelChecks.Add(
                        "The model does not contain a Visual Observation Placeholder Input " +
                        "for visual observation " + visObsIndex + ".");
                }
            }

            // If the model has a non-negative memory size but requires a recurrent input
            if (memory > 0)
            {
                if (!tensorsNames.Any(x => x.EndsWith("_h")) ||
                    !tensorsNames.Any(x => x.EndsWith("_c")))
                {
                    m_FailedModelChecks.Add(
                        "The model does not contain a Recurrent Input Node but has memory_size.");
                }
            }

            // If the model uses discrete control but does not have an input for action masks
            if (isContinuous == ModelActionType.Discrete)
            {
                if (!tensorsNames.Contains(TensorNames.ActionMaskPlaceholder))
                {
                    m_FailedModelChecks.Add(
                        "The model does not contain an Action Mask but is using Discrete Control.");
                }
            }
        }

        /// <summary>
        /// Generates failed checks that correspond to outputs expected by the model that are not
        /// present in the BrainParameters.
        /// </summary>
        /// <param name="memory">The memory size that the model is expecting/</param>
        /// <returns>
        /// A IEnumerable of string corresponding to the failed output presence checks.
        /// </returns>
        private void CheckOutputTensorPresence(int memory)
        {
            // If there is no Action Output.
            if (!m_Model.outputs.Contains(TensorNames.ActionOutput))
            {
                m_FailedModelChecks.Add("The model does not contain an Action Output Node.");
            }

            // If there is no Recurrent Output but the model is Recurrent.
            if (memory > 0)
            {
                var memOutputs = m_Model.memories.Select(x => x.output).ToList();

                if (!memOutputs.Any(x => x.EndsWith("_h")) ||
                    !memOutputs.Any(x => x.EndsWith("_c")))
                {
                    m_FailedModelChecks.Add(
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
                new Dictionary<string, Func<TensorProxy, string>>()
            {
                {TensorNames.VectorObservationPlacholder, CheckVectorObsShape},
                {TensorNames.PreviousActionPlaceholder, CheckPreviousActionShape},
                {TensorNames.RandomNormalEpsilonPlaceholder, ((tensor) => null)},
                {TensorNames.ActionMaskPlaceholder, ((tensor) => null)},
                {TensorNames.SequenceLengthPlaceholder, ((tensor) => null)},
                {TensorNames.RecurrentInPlaceholder, ((tensor) => null)},
            };

            foreach (var mem in m_Model.memories)
            {
                tensorTester[mem.input] = ((tensor) => null);
            }

            for (var obsIndex = 0; obsIndex < m_BrainParameters.cameraResolutions.Length; obsIndex++)
            {
                var index = obsIndex;
                tensorTester[TensorNames.VisualObservationPlaceholderPrefix + obsIndex] =
                    (tensor) => CheckVisualObsShape(tensor, index);
            }
            // If the model expects an input but it is not in this list
            foreach (var tensor in GetInputTensors())
            {
                if (!tensorTester.ContainsKey(tensor.name))
                {
                    m_FailedModelChecks.Add(
                        "Model requires an unknown input named : " + tensor.name);
                }
                else
                {
                    var tester = tensorTester[tensor.name];
                    var error = tester.Invoke(tensor);
                    if (error != null)
                    {
                        m_FailedModelChecks.Add(error);
                    }
                }
            }
        }

        /// <summary>
        /// Checks that the shape of the Vector Observation input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        private string CheckVectorObsShape(TensorProxy tensorProxy)
        {
            var vecObsSizeBp = m_BrainParameters.vectorObservationSize;
            var numStackedVector = m_BrainParameters.numStackedVectorObservations;
            var totalVecObsSizeT = tensorProxy.shape[tensorProxy.shape.Length - 1];
            if (vecObsSizeBp * numStackedVector != totalVecObsSizeT)
            {
                return "Vector Observation Size of the model does not match. Received " +
                    $"{vecObsSizeBp} x {numStackedVector} but was expecting {totalVecObsSizeT}.";
            }
            return null;
        }

        /// <summary>
        /// Checks that the shape of the Previous Vector Action input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensorProxy"> The tensor that is expected by the model</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckPreviousActionShape(TensorProxy tensorProxy)
        {
            var numberActionsBp = m_BrainParameters.vectorActionSize.Length;
            var numberActionsT = tensorProxy.shape[tensorProxy.shape.Length - 1];
            if (numberActionsBp != numberActionsT)
            {
                return "Previous Action Size of the model does not match. " +
                    $"Received {numberActionsBp} but was expecting {numberActionsT}.";
            }
            return null;
        }

        /// <summary>
        /// Checks that the shape of the visual observation input placeholder is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="tensorProxy">The tensor that is expected by the model</param>
        /// <param name="visObsIndex">The index of the visual observation.</param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        private string CheckVisualObsShape(TensorProxy tensorProxy, int visObsIndex)
        {
            var resolutionBp = m_BrainParameters.cameraResolutions[visObsIndex];
            var widthBp = resolutionBp.width;
            var heightBp = resolutionBp.height;
            var pixelBp = resolutionBp.blackAndWhite ? 1 : 3;
            var heightT = tensorProxy.shape[1];
            var widthT = tensorProxy.shape[2];
            var pixelT = tensorProxy.shape[3];
            if ((widthBp != widthT) || (heightBp != heightT) || (pixelBp != pixelT))
            {
                return $"The visual Observation {visObsIndex} of the model does not match. " +
                    $"Received TensorProxy of shape [?x{widthBp}x{heightBp}x{pixelBp}] but " +
                    $"was expecting [?x{widthT}x{heightT}x{pixelT}].";
            }
            return null;
        }

        /// <summary>
        /// Generates failed checks that correspond to output shapes incompatibilities between
        /// the model and the BrainParameters.
        /// </summary>
        /// <param name="isContinuous">
        /// Whether the model is expecting continuous or discrete control.
        /// </param>
        /// <param name="modelActionSize">
        /// The size of the action output that is expected by the model.
        /// </param>
        /// <returns>
        /// A IEnumerable of string corresponding to the incompatible shapes between model
        /// and BrainParameters.
        /// </returns>
        private void CheckOutputTensorShape(ModelActionType isContinuous, int modelActionSize)
        {
            if (isContinuous == ModelActionType.Unknown)
            {
                m_FailedModelChecks.Add("Cannot infer type of Control from the provided model.");
                return;
            }
            if (isContinuous == ModelActionType.Continuous &&
                m_BrainParameters.vectorActionSpaceType != SpaceType.Continuous)
            {
                m_FailedModelChecks.Add(
                    "Model has been trained using Continuous Control but the Brain Parameters " +
                    "suggest Discrete Control.");
                return;
            }
            if (isContinuous == ModelActionType.Discrete &&
                m_BrainParameters.vectorActionSpaceType != SpaceType.Discrete)
            {
                m_FailedModelChecks.Add(
                    "Model has been trained using Discrete Control but the Brain Parameters " +
                    "suggest Continuous Control.");
                return;
            }
            var tensorTester = new Dictionary<string, Func<TensorShape, int, string>>();
            if (m_BrainParameters.vectorActionSpaceType == SpaceType.Continuous)
            {
                tensorTester[TensorNames.ActionOutput] = CheckContinuousActionOutputShape;
            }
            else
            {
                tensorTester[TensorNames.ActionOutput] = CheckDiscreteActionOutputShape;
            }
            // If the model expects an output but it is not in this list
            foreach (var name in m_Model.outputs)
            {
                if (tensorTester.ContainsKey(name))
                {
                    var tester = tensorTester[name];
                    var error = tester.Invoke(m_Model.GetShapeByName(name), modelActionSize);
                    if (error != null)
                    {
                        m_FailedModelChecks.Add(error);
                    }
                }
            }
        }

        /// <summary>
        /// Checks that the shape of the discrete action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="shape"> The tensor shape that is expected by the model</param>
        /// <param name="modelActionSize">
        /// The size of the action output that is expected by the model.
        /// </param>
        /// <returns>
        /// If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.
        /// </returns>
        private string CheckDiscreteActionOutputShape(TensorShape shape, int modelActionSize)
        {
            var bpActionSize = m_BrainParameters.vectorActionSize.Sum();
            if (modelActionSize != bpActionSize)
            {
                return "Action Size of the model does not match. The BrainParameters expect " +
                    $"{bpActionSize} but the model contains {modelActionSize}.";
            }
            return null;
        }

        /// <summary>
        /// Checks that the shape of the continuous action output is the same in the
        /// model and in the Brain Parameters.
        /// </summary>
        /// <param name="shape"> The tensor shape that is expected by the model</param>
        /// <param name="modelActionSize">
        /// The size of the action output that is expected by the model.
        /// </param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckContinuousActionOutputShape(TensorShape shape, int modelActionSize)
        {
            var bpActionSize = m_BrainParameters.vectorActionSize[0];
            if (modelActionSize != bpActionSize)
            {
                return "Action Size of the model does not match. The BrainParameters expect " +
                    $"{bpActionSize} but the model contains {modelActionSize}.";
            }
            return null;
        }
    }
}
