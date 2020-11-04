using Unity.Barracuda;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Barracuda Model extension methods.
    /// </summary>
    internal static class BarracudaModelExtensions
    {
        /// <summary>
        /// Check if the model has continuous action outputs.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>True if the model has continuous action outputs.</returns>
        public static bool HasContinuousOutputs(this Model model)
        {
            if (model.UseDeprecated())
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
            if (model.UseDeprecated())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    (int)model.GetTensorByName(TensorNames.ActionOutputShapeDeprecated)[0] : 0;
            }
            else
            {
                return (int)model.GetTensorByName(TensorNames.ContinuousActionOutputShape)[0];
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
            if (model.UseDeprecated())
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
            if (model.UseDeprecated())
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
        /// Discrete action output size of the model.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>Size of discrete action output.</returns>
        public static int DiscreteOutputSize(this Model model)
        {
            if (model.UseDeprecated())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    0 : (int)model.GetTensorByName(TensorNames.ActionOutputShapeDeprecated)[0];
            }
            else
            {
                return (int)model.GetTensorByName(TensorNames.DiscreteActionOutputShape)[0];
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
            if (model.UseDeprecated())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return TensorNames.DiscreteActionOutput;
            }
        }

        /// <summary>
        /// Check if the model uses deprecated output fields and should be handled differently.
        /// </summary>
        /// <param name="model">
        /// The Barracuda engine model for loading static parameters.
        /// </param>
        /// <returns>True if the model uses deprecated output fields.</returns>
        public static bool UseDeprecated(this Model model)
        {
            return !model.outputs.Contains(TensorNames.ContinuousActionOutput) &&
                !model.outputs.Contains(TensorNames.DiscreteActionOutput);
        }
    }
}
