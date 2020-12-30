using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Barracuda Model extension methods.
    /// </summary>
    internal static class BarracudaModelExtensions
    {
        /// <summary>
        /// Get array of the input tensor names of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Array of the input tensor names of the model</returns>
        public static string[] GetInputNames(this Model model)
        {
            var names = new List<string>();

            if (model == null)
                return names.ToArray();

            foreach (var input in model.inputs)
            {
                names.Add(input.name);
            }

            foreach (var mem in model.memories)
            {
                names.Add(mem.input);
            }

            names.Sort();

            return names.ToArray();
        }

        /// <summary>
        /// Generates the Tensor inputs that are expected to be present in the Model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>TensorProxy IEnumerable with the expected Tensor inputs.</returns>
        public static IReadOnlyList<TensorProxy> GetInputTensors(this Model model)
        {
            var tensors = new List<TensorProxy>();

            if (model == null)
                return tensors;

            foreach (var input in model.inputs)
            {
                tensors.Add(new TensorProxy
                {
                    name = input.name,
                    valueType = TensorProxy.TensorType.FloatingPoint,
                    data = null,
                    shape = input.shape.Select(i => (long)i).ToArray()
                });
            }

            foreach (var mem in model.memories)
            {
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
        /// Get number of visual observation inputs to the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Number of visual observation inputs to the model</returns>
        public static int GetNumVisualInputs(this Model model)
        {
            var count = 0;
            if (model == null)
                return count;

            foreach (var input in model.inputs)
            {
                if (input.name.StartsWith(TensorNames.VisualObservationPlaceholderPrefix))
                {
                    count++;
                }
            }

            return count;
        }

        /// <summary>
        /// Get array of the output tensor names of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Array of the output tensor names of the model</returns>
        public static string[] GetOutputNames(this Model model)
        {
            var names = new List<string>();

            if (model == null)
            {
                return names.ToArray();
            }

            if (model.HasContinuousOutputs())
            {
                names.Add(model.ContinuousOutputName());
            }
            if (model.HasDiscreteOutputs())
            {
                names.Add(model.DiscreteOutputName());
            }

            var memory = (int)model.GetTensorByName(TensorNames.MemorySize)[0];
            if (memory > 0)
            {
                foreach (var mem in model.memories)
                {
                    names.Add(mem.output);
                }
            }

            names.Sort();

            return names.ToArray();
        }

        /// <summary>
        /// Check if the model has continuous action outputs.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>True if the model has continuous action outputs.</returns>
        public static bool HasContinuousOutputs(this Model model)
        {
            if (model == null)
                return false;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0;
            }
            else
            {
                return model.outputs.Contains(TensorNames.ContinuousActionOutput) &&
                    (int)model.GetTensorByName(TensorNames.ContinuousActionOutputShape)[0] > 0;
            }
        }

        /// <summary>
        /// Continuous action output size of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Size of continuous action output.</returns>
        public static int ContinuousOutputSize(this Model model)
        {
            if (model == null)
                return 0;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    (int)model.GetTensorByName(TensorNames.ActionOutputShapeDeprecated)[0] : 0;
            }
            else
            {
                var continuousOutputShape = model.GetTensorByName(TensorNames.ContinuousActionOutputShape);
                return continuousOutputShape == null ? 0 : (int)continuousOutputShape[0];
            }
        }

        /// <summary>
        /// Continuous action output tensor name of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Tensor name of continuous action output.</returns>
        public static string ContinuousOutputName(this Model model)
        {
            if (model == null)
                return null;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return TensorNames.ContinuousActionOutput;
            }
        }

        /// <summary>
        /// Check if the model has discrete action outputs.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>True if the model has discrete action outputs.</returns>
        public static bool HasDiscreteOutputs(this Model model)
        {
            if (model == null)
                return false;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] == 0;
            }
            else
            {
                return model.outputs.Contains(TensorNames.DiscreteActionOutput) &&
                    (int)model.GetTensorByName(TensorNames.DiscreteActionOutputShape)[0] > 0;
            }
        }

        /// <summary>
        /// Discrete action output size of the model. This is equal to the sum of the branch sizes.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Size of discrete action output.</returns>
        public static int DiscreteOutputSize(this Model model)
        {
            if (model == null)
                return 0;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    0 : (int)model.GetTensorByName(TensorNames.ActionOutputShapeDeprecated)[0];
            }
            else
            {
                var discreteOutputShape = model.GetTensorByName(TensorNames.DiscreteActionOutputShape);
                return discreteOutputShape == null ? 0 : (int)discreteOutputShape[0];
            }
        }

        /// <summary>
        /// Discrete action output tensor name of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Tensor name of discrete action output.</returns>
        public static string DiscreteOutputName(this Model model)
        {
            if (model == null)
                return null;
            if (!model.SupportsContinuousAndDiscrete())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return TensorNames.DiscreteActionOutput;
            }
        }

        /// <summary>
        /// Check if the model supports both continuous and discrete actions.
        /// If not, the model should be handled differently and use the deprecated fields.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>True if the model supports both continuous and discrete actions.</returns>
        public static bool SupportsContinuousAndDiscrete(this Model model)
        {
            return model == null ||
                model.outputs.Contains(TensorNames.ContinuousActionOutput) ||
                model.outputs.Contains(TensorNames.DiscreteActionOutput);
        }

        /// <summary>
        /// Check if the model contains all the expected input/output tensors.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <param name="failedModelChecks">Output list of failure messages</param>
        ///
        /// <returns>True if the model contains all the expected tensors.</returns>
        public static bool CheckExpectedTensors(this Model model, List<string> failedModelChecks)
        {
            // Check the presence of model version
            var modelApiVersionTensor = model.GetTensorByName(TensorNames.VersionNumber);
            if (modelApiVersionTensor == null)
            {
                failedModelChecks.Add($"Required constant \"{TensorNames.VersionNumber}\" was not found in the model file.");
                return false;
            }

            // Check the presence of memory size
            var memorySizeTensor = model.GetTensorByName(TensorNames.MemorySize);
            if (memorySizeTensor == null)
            {
                failedModelChecks.Add($"Required constant \"{TensorNames.MemorySize}\" was not found in the model file.");
                return false;
            }

            // Check the presence of action output tensor
            if (!model.outputs.Contains(TensorNames.ActionOutputDeprecated) &&
                !model.outputs.Contains(TensorNames.ContinuousActionOutput) &&
                !model.outputs.Contains(TensorNames.DiscreteActionOutput))
            {
                failedModelChecks.Add("The model does not contain any Action Output Node.");
                return false;
            }

            // Check the presence of action output shape tensor
            if (!model.SupportsContinuousAndDiscrete())
            {
                if (model.GetTensorByName(TensorNames.ActionOutputShapeDeprecated) == null)
                {
                    failedModelChecks.Add("The model does not contain any Action Output Shape Node.");
                    return false;
                }
                if (model.GetTensorByName(TensorNames.IsContinuousControlDeprecated) == null)
                {
                    failedModelChecks.Add($"Required constant \"{TensorNames.IsContinuousControlDeprecated}\" was not found in the model file. " +
                        "This is only required for model that uses a deprecated model format.");
                    return false;
                }
            }
            else
            {
                if (model.outputs.Contains(TensorNames.ContinuousActionOutput) &&
                    model.GetTensorByName(TensorNames.ContinuousActionOutputShape) == null)
                {
                    failedModelChecks.Add("The model uses continuous action but does not contain Continuous Action Output Shape Node.");
                    return false;
                }
                if (model.outputs.Contains(TensorNames.DiscreteActionOutput) &&
                    model.GetTensorByName(TensorNames.DiscreteActionOutputShape) == null)
                {
                    failedModelChecks.Add("The model uses discrete action but does not contain Discrete Action Output Shape Node.");
                    return false;
                }
            }
            return true;
        }
    }
}
