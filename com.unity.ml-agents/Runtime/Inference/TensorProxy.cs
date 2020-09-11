using System;
using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Inference.Utils;

namespace Unity.MLAgents.Inference
{
    /// <summary>
    /// Tensor - A class to encapsulate a Tensor used for inference.
    ///
    /// This class contains the Array that holds the data array, the shapes, type and the
    /// placeholder in the execution graph. All the fields are editable in the inspector,
    /// allowing the user to specify everything but the data in a graphical way.
    /// </summary>
    [Serializable]
    internal class TensorProxy
    {
        public enum TensorType
        {
            Integer,
            FloatingPoint
        };

        static readonly Dictionary<TensorType, Type> k_TypeMap =
            new Dictionary<TensorType, Type>()
        {
            {TensorType.FloatingPoint, typeof(float)},
            {TensorType.Integer, typeof(int)}
        };

        public string name;
        public TensorType valueType;

        // Since Type is not serializable, we use the DisplayType for the Inspector
        public Type DataType => k_TypeMap[valueType];
        public long[] shape;
        public Tensor data;

        public long Height
        {
            get { return shape.Length == 4 ? shape[1] : shape[5]; }
        }

        public long Width
        {
            get { return shape.Length == 4 ? shape[2] : shape[6]; }
        }

        public long Channels
        {
            get { return shape.Length == 4 ? shape[3] : shape[7]; }
        }
    }

    internal static class TensorUtils
    {
        public static void ResizeTensor(TensorProxy tensor, int batch, ITensorAllocator allocator)
        {
            if (tensor.shape[0] == batch &&
                tensor.data != null && tensor.data.batch == batch)
            {
                return;
            }

            tensor.data?.Dispose();
            tensor.shape[0] = batch;

            if (tensor.shape.Length == 4 || tensor.shape.Length == 8)
            {
                tensor.data = allocator.Alloc(
                    new TensorShape(
                        batch,
                        (int)tensor.Height,
                        (int)tensor.Width,
                        (int)tensor.Channels));
            }
            else
            {
                tensor.data = allocator.Alloc(
                    new TensorShape(
                        batch,
                        (int)tensor.shape[tensor.shape.Length - 1]));
            }
        }

        internal static long[] TensorShapeFromBarracuda(TensorShape src)
        {
            if (src.height == 1 && src.width == 1)
            {
                return new long[] { src.batch, src.channels };
            }

            return new long[] { src.batch, src.height, src.width, src.channels };
        }

        public static TensorProxy TensorProxyFromBarracuda(Tensor src, string nameOverride = null)
        {
            var shape = TensorShapeFromBarracuda(src.shape);
            return new TensorProxy
            {
                name = nameOverride ?? src.name,
                valueType = TensorProxy.TensorType.FloatingPoint,
                shape = shape,
                data = src
            };
        }

        /// <summary>
        /// Fill a specific batch of a TensorProxy with a given value
        /// </summary>
        /// <param name="tensorProxy"></param>
        /// <param name="batch">The batch index to fill.</param>
        /// <param name="fillValue"></param>
        public static void FillTensorBatch(TensorProxy tensorProxy, int batch, float fillValue)
        {
            var height = tensorProxy.data.height;
            var width = tensorProxy.data.width;
            var channels = tensorProxy.data.channels;
            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        tensorProxy.data[batch, h, w, c] = fillValue;
                    }
                }
            }
        }

        /// <summary>
        /// Fill a pre-allocated Tensor with random numbers
        /// </summary>
        /// <param name="tensorProxy">The pre-allocated Tensor to fill</param>
        /// <param name="randomNormal">RandomNormal object used to populate tensor</param>
        /// <exception cref="NotImplementedException">
        /// Throws when trying to fill a Tensor of type other than float
        /// </exception>
        /// <exception cref="ArgumentNullException">
        /// Throws when the Tensor is not allocated
        /// </exception>
        public static void FillTensorWithRandomNormal(
            TensorProxy tensorProxy, RandomNormal randomNormal)
        {
            if (tensorProxy.DataType != typeof(float))
            {
                throw new NotImplementedException("Only float data types are currently supported");
            }

            if (tensorProxy.data == null)
            {
                throw new ArgumentNullException();
            }

            for (var i = 0; i < tensorProxy.data.length; i++)
            {
                tensorProxy.data[i] = (float)randomNormal.NextDouble();
            }
        }
    }
}
