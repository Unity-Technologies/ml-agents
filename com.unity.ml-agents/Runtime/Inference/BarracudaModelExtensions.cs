using Unity.Barracuda;
using UnityEngine;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Barracuda Model extension methods.
    /// </summary>
    internal static class BarracudaModelExtensions
    {
        public static bool HasContinuousOutputs(this Model model)
        {
            Debug.Log("HasContinuousOutputs");
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

        public static int ContinuousOutputSize(this Model model)
        {
            Debug.Log("ContinuousOutputSize");
            if (model.UseDeprecated())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    (int)model.GetTensorByName(TensorNames.ActionOutputDeprecated)[0] : 0;
            }
            else
            {
                return (int)model.GetTensorByName(TensorNames.ContinuousActionOutputShape)[0];
            }
        }

        public static string ContinuousOutputName(this Model model)
        {
            Debug.Log("ContinuousOutputName");
            if (model.UseDeprecated())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return TensorNames.ContinuousActionOutput;
            }
        }

        public static bool HasDiscreteOutputs(this Model model)
        {
            Debug.Log("HasDiscreteOutput");
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

        public static int DiscreteOutputSize(this Model model)
        {
            Debug.Log("DiscreteOutputSize");
            if (model.UseDeprecated())
            {
                return (int)model.GetTensorByName(TensorNames.IsContinuousControlDeprecated)[0] > 0 ?
                    0 : (int)model.GetTensorByName(TensorNames.ActionOutputDeprecated)[0];
            }
            else
            {
                return (int)model.GetTensorByName(TensorNames.DiscreteActionOutputShape)[0];
            }
        }

        public static string DiscreteOutputName(this Model model)
        {
            Debug.Log("DiscreteOutputName");
            if (model.UseDeprecated())
            {
                return TensorNames.ActionOutputDeprecated;
            }
            else
            {
                return TensorNames.DiscreteActionOutput;
            }
        }

        private static bool UseDeprecated(this Model model)
        {
            return !model.outputs.Contains(TensorNames.ContinuousActionOutput) &&
                !model.outputs.Contains(TensorNames.DiscreteActionOutput);
        }
    }
}
