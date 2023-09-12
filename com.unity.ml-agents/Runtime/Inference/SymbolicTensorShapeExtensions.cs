using Unity.Sentis;
using UnityEngine.Assertions;

namespace Unity.MLAgents.Inference
{
    internal static class SymbolicTensorShapeExtensions
    {
        public static long[] ToArray(this SymbolicTensorShape shape)
        {
            var shapeOut = new long[shape.rank];

            Assert.IsTrue(shape.IsFullyKnown(), "ValueError: Cannot convert tensor of unknown rank to TensorShape");

            for (var i = 0; i < shape.rank; i++)
            {
                if (shape[i].isParam)
                {
                    shapeOut[i] = 1;
                }
                else
                {
                    shapeOut[i] = shape[i].value;
                }
            }

            return shapeOut;
        }
    }
}
