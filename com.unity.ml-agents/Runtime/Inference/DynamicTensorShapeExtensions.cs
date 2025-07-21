using Unity.InferenceEngine;
using UnityEngine.Assertions;

namespace Unity.MLAgents.Inference
{
    static class DynamicTensorShapeExtensions
    {
        public static int[] ToArray(this DynamicTensorShape shape)
        {
            var shapeOut = new int[shape.rank];

            // TODO investigate how critical this is and if we can just remove this assert. the alternative is to expose this again in Sentis.

            // Assert.IsTrue(shape.hasRank, "ValueError: Cannot convert tensor of unknown rank to TensorShape");

            var shapeArray = shape.ToIntArray();

            for (var i = 0; i < shape.rank; i++)
            {
                if (shapeArray[i] == -1)
                {
                    shapeOut[i] = 1;
                }
                else
                {
                    shapeOut[i] = shapeArray[i];
                }
            }

            return shapeOut;
        }
    }
}
