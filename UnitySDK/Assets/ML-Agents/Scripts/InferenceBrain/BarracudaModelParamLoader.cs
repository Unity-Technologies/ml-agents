#define ENABLE_BARRACUDA
#if ENABLE_BARRACUDA
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Barracuda;
using UnityEngine;
using Tensor = MLAgents.InferenceBrain.Tensor;

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
        private const long ApiVersion = 2;
        private IWorker _engine;
        private Model _model;
        private BrainParameters _brainParameters;
        private List<string> _failedModelChecks = new List<string>();

        /// <summary>
        /// Factory for the ModelParamLoader : Creates a ModelParamLoader and runs the checks
        /// on it.
        /// </summary>
        /// <param name="engine"> The Barracuda engine worker we get the parameters and the checks from
        /// </param>
        /// <param name="model"> The Barracuda engine model for loading static parameters
        /// </param>
        /// <param name="brainParameters"> The BrainParamters that are used verify the
        /// compatibility with the InferenceEngine</param>
        /// <returns></returns>
        public static BarracudaModelParamLoader GetLoaderAndCheck(IWorker engine, Model model,
            BrainParameters brainParameters)
        {
            BarracudaModelParamLoader modelParamLoader = new BarracudaModelParamLoader(engine, model, brainParameters);
            modelParamLoader.GenerateChecks();
            return modelParamLoader;
        }
        
        private BarracudaModelParamLoader(IWorker engine, Model model, BrainParameters brainParameters)
        {
            _engine = engine;
            _model = model;
            _brainParameters = brainParameters;
        }

        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model. 
        /// </summary>
        /// <returns>Tensor IEnumerable with the expected Tensor inputs</returns>
        public IReadOnlyList<Tensor> GetInputTensors()
        {
            List<Tensor> tensors = new List<Tensor>();

            if (_model == null)
                return tensors;
            
            foreach (var input in _model.inputs)
            {
                tensors.Add(new Tensor
                {
                    Name = input.name,
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null,
                    Shape = input.shape.Select(i => (long)i).ToArray()
                });
            }
            
            foreach (var mem in _model.memories)
            {
                //Debug.Log($"{mem.input}: {mem.shape} -> {BarracudaUtils.FromBarracuda(mem.shape).Length}");
                tensors.Add(new Tensor
                {
                    Name = mem.input,
                    ValueType = Tensor.TensorType.FloatingPoint,
                    Data = null,
                    Shape = BarracudaUtils.FromBarracuda(mem.shape)
                });
            }
            
            tensors.Sort((el1, el2) => el1.Name.CompareTo(el2.Name));
            
            return tensors;
        }
        
        /// <summary>
        /// Generates the Tensor outputs that are expected to be present in the Model. 
        /// </summary>
        /// <returns>Tensor IEnumerable with the expected Tensor outputs</returns>
        public string[] GetOutputNames()
        {
            var names = new List<string>();

            if (_model == null)
                return names.ToArray();
            
            names.Add(TensorNames.ActionOutput);                
             
            var memory = GetIntScalar(TensorNames.MemorySize);
            if (memory > 0)
            {
                names.Add(TensorNames.RecurrentOutput_C);
                names.Add(TensorNames.RecurrentOutput_H);
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
            return (int)_model.GetTensorByName(name)[0];
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
                if (!tensorsNames.Contains(TensorNames.RecurrentInPlaceholder_H) ||
                    !tensorsNames.Contains(TensorNames.RecurrentInPlaceholder_C))
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
            // If there is no Action Output.
            if (!_model.outputs.Contains(TensorNames.ActionOutput))
            {
                _failedModelChecks.Add("The model does not contain an Action Output Node.");
            }
            
            // If there is no Recurrent Output but the model is Recurrent.
            if (memory > 0)
            {
                var memOutputs = _model.memories.Select(x => x.output).ToList();
                
                if (!memOutputs.Contains(TensorNames.RecurrentOutput_H) || 
                    !memOutputs.Contains(TensorNames.RecurrentOutput_C))
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
                    {TensorNames.RecurrentInPlaceholder_H, ((tensor) => null)},
                    {TensorNames.RecurrentInPlaceholder_C, ((tensor) => null)},
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
            var totalVecObsSizeT = tensor.Shape[tensor.Shape.Length - 1];
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
            var numberActionsT = tensor.Shape[tensor.Shape.Length - 1];
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
            var tensorTester = new Dictionary<string, Func<TensorShape, int, string>>();
            if (_brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                tensorTester[TensorNames.ActionOutput] = CheckContinuousActionOutputShape;
            }
            else
            {
                tensorTester[TensorNames.ActionOutput] = CheckDiscreteActionOutputShape;
            }
            // If the model expects an output but it is not in this list
            foreach (var name in _model.outputs)
            {
                if (tensorTester.ContainsKey(name))
                {
                    var tester = tensorTester[name];
                    var error = tester.Invoke(_model.GetShapeByName(name), modelActionSize);
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
        /// <param name="shape"> The tensor shape that is expected by the model</param>
        /// <param name="modelActionSize"> The size of the action output that is expected
        /// by the model.</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckDiscreteActionOutputShape(TensorShape shape, int modelActionSize)
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
        /// <param name="shape"> The tensor shape that is expected by the model</param>
        /// <param name="modelActionSize"> The size of the action output that is expected
        /// by the model.</param>
        /// <returns>If the Check failed, returns a string containing information about why the
        /// check failed. If the check passed, returns null.</returns>
        private string CheckContinuousActionOutputShape(TensorShape shape, int modelActionSize)
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

public class BarracudaUtils
{
    private static Array LinearizeArray(Array src)  
    {
        var elementType = src.GetType().GetElementType();
        var elementSize = Marshal.SizeOf(elementType);
        var dest = Array.CreateInstance(elementType, src.Length);
        Buffer.BlockCopy(src, 0, dest, 0, src.Length * elementSize);
        return dest;
    }
    
    protected static Barracuda.TensorShape ToBarracuda(long[] src)
    {
        if (src.Length > 4)
            throw new NotImplementedException("Barracuda does not support Tensor shapes with rank higher than 4");

        var shape = new int[4];

        if (src.Length == 2)
        {
            shape[0] = (int)src[0];
            shape[1] = 1;
            shape[2] = 1;
            shape[3] = (int)src[1];
        }
        else
        {
            for (var axis = 0; axis < src.Length; ++axis)
                shape[shape.Length-axis-1] = (int)src[src.Length-axis-1];
        }
        
        return new Barracuda.TensorShape(shape);
    }
    
    private static float[] IntArrayToFloatArray(int[] src)
    {
        var dest = new float[src.Length];
        for (var i = 0; i < src.Length; i++)
            dest[i] = (float) src[i];

        return dest;
    }
    
    public static Barracuda.Tensor ToBarracuda(MLAgents.InferenceBrain.Tensor src)
    {
        Array linearArray = LinearizeArray(src.Data);

        if (linearArray.GetType().GetElementType() == typeof(int))
            linearArray = IntArrayToFloatArray(linearArray as int[]);

        var shape = ToBarracuda(src.Shape);
        return new Barracuda.Tensor(shape,  linearArray as float[], src.Name);
    }
    
    internal static long[] FromBarracuda(Barracuda.TensorShape src)
    {
        if (src.height == 1 && src.width == 1)
            return new long[2] {src.batch, src.channels};

        return new long[4] {src.batch, src.height, src.width, src.channels};
    }
    
    private static Array ReshapeArray(Array src, long[] shape)
    {
        var elementType = src.GetType().GetElementType();
        var elementSize = Marshal.SizeOf(elementType);
        var dest = Array.CreateInstance(elementType, shape);
        Buffer.BlockCopy(src, 0, dest, 0, src.Length * elementSize);
        return dest;
    }
    
    public static Tensor FromBarracuda(Barracuda.Tensor src, string nameOverride = null)
    {
        var shape = FromBarracuda(src.shape);
        return new Tensor
        {
            Name = nameOverride ?? src.name,
            ValueType = Tensor.TensorType.FloatingPoint,
            Shape = shape,
            Data = ReshapeArray(src.data.Download(src.length), shape)
        };
    }
}
#endif
