using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using Unity.MLAgents.Inference.Utils;
using Unity.MLAgents.Policies;

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
            { TensorType.FloatingPoint, typeof(float) },
            { TensorType.Integer, typeof(int) }
        };

        static readonly Dictionary<TensorType, DataType> k_DTypeMap =
            new Dictionary<TensorType, DataType>()
        {
            { TensorType.FloatingPoint, Sentis.DataType.Float },
            { TensorType.Integer, Sentis.DataType.Int }
        };

        public string name;
        public TensorType valueType;

        // Since Type is not serializable, we use the DisplayType for the Inspector
        public Type DataType => k_TypeMap[valueType];
        public DataType DType => k_DTypeMap[valueType];
        public long[] shape;
        public Tensor data;
        public DeviceType Device => data.tensorOnDevice.deviceType;

        public long Height
        {
            get { return shape.Length >= 4 ? shape[^2] : 1; }
        }

        public long Width
        {
            get { return shape.Length >= 3 ? shape[^1] : 1; }
        }

        public long Channels
        {
            get
            {
                return shape.Length >= 4 ? shape[^3] :
                    shape.Length == 3 ? shape[^2] :
                    shape.Length == 2 ? shape[^1] : 1;
            }
        }

        ~TensorProxy()
        {
            data?.Dispose();
        }
    }

    internal static class TensorUtils
    {
        public static void ResizeTensor(TensorProxy tensor, int batch, ITensorAllocator allocator)
        {
            if (tensor.shape[0] == batch &&
                tensor.data != null && tensor.data.Batch() == batch)
            {
                return;
            }

            tensor.data?.Dispose();
            tensor.shape[0] = batch;
            var newTensorShape = new TensorShape(tensor.shape.Select(i => (int)i).ToArray());
            tensor.data = CreateEmptyTensor(newTensorShape, tensor.DType);
        }

        public static Tensor CreateEmptyTensor(TensorShape shape, DataType dataType)
        {
            Tensor tensor = null;
            switch (dataType)
            {
                case DataType.Float:
                    tensor = TensorFloat.Zeros(shape);
                    break;
                case DataType.Int:
                    tensor = TensorInt.Zeros(shape);
                    break;
            }

            return tensor;
        }

        internal static long[] TensorShapeFromSentis(TensorShape src)
        {
            if (src.rank == 2)
            {
                return new long[] { src.Batch(), src.Channels() };
            }

            if (src.Height() == 1 && src.Width() == 1)
            {
                return new long[] { src.Batch(), src.Channels() };
            }

            return new long[] { src.Batch(), src.Channels(), src.Height(), src.Width() };
        }

        public static TensorProxy TensorProxyFromSentis(Tensor src, string nameOverride = null)
        {
            var shape = TensorShapeFromSentis(src.shape);
            return new TensorProxy
            {
                // name = nameOverride ?? src.name,
                name = nameOverride ?? "",
                valueType = src.dataType == DataType.Float
                    ? TensorProxy.TensorType.FloatingPoint
                    : TensorProxy.TensorType.Integer,
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
            var height = tensorProxy.data.Height();
            var width = tensorProxy.data.Width();
            var channels = tensorProxy.data.Channels();
            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        ((TensorFloat)tensorProxy.data)[batch, c, h, w] = fillValue;
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

            for (var i = 0; i < tensorProxy.data.Length(); i++)
            {
                ((TensorFloat)tensorProxy.data)[i] = (float)randomNormal.NextDouble();
            }
        }
    }
}
