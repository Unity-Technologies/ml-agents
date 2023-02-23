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

        public const int ChIndex = 0;
        public const int HIndex = 1;
        public const int WIndex = 2;

        public string name;
        public DataType valueType;

        // Since Type is not serializable, we use the DisplayType for the Inspector
        public int[] shape;
        public Tensor data;

        public TensorFloat FloatData => data as TensorFloat;
        //public TensorFloat FloatData => valueType == DataType.Float ? data as TensorFloat : null;
        public TensorInt IntData => data as TensorInt;
        //public TensorInt IntData => valueType == DataType.Float ? null : data as TensorInt;


        ////@Barracude4Upgrade:  Old Height, Width, Channels accessors go away.  Possibly TensorProxy could go completely?

    }

    internal static class TensorUtils
    {
        public static void ResizeTensor(TensorProxy tensor, int batch, ITensorAllocator allocator)
        {
            if (tensor.shape[TensorProxy.ChIndex] == batch &&            
                tensor.data != null && tensor.data.shape[TensorProxy.ChIndex] == batch)  //@TODO: verify correctness
            {
                return;
            }

            tensor.data?.Dispose();
            tensor.shape[TensorProxy.ChIndex] = batch;

            if (tensor.valueType == DataType.Float)
            {
                tensor.data = allocator.Alloc(new TensorShape(tensor.shape), DataType.Float) as TensorFloat;
            }
            else
            {
                tensor.data = allocator.Alloc(new TensorShape(tensor.shape), DataType.Int) as TensorInt;
            }
        }


        internal static int[] TensorShapeFromBarracuda(TensorShape src)
        {
            //@Barracude4Upgrade:
            // //@TODO: remove this since ToArray handles it the same way.
            // if (src.rank == 2)
            // //if (src[-3] == 1 && src[-2] == 1) //@TODO: verify correctness
            // {
            //     return new int[] { src[0], src[1] };  //@TODO: verify correctness
            // }

            return src.ToArray();  //new int[] { src.batch, src.height, src.width, src.channels };  ////@TODO: verify correctness
        }

        public static TensorProxy TensorProxyFromBarracuda(Tensor src, string nameOverride = null)
        {
            var shape = TensorShapeFromBarracuda(src.shape);
            return new TensorProxy
            {
                name = nameOverride ?? src.name,
                //@Barracude4Upgrade:
                valueType = src.dataType,
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
            //@Barracude4Upgrade: There's likely a more efficient way?

            // var height = tensorProxy.data.shape[-3]; //height;  //@TODO: verify correctness
            // var width = tensorProxy.data.shape[-2]; //width;  //@TODO: verify correctness
            // var channels = tensorProxy.data.shape[-1]; //channels; //@TODO: verify correctness
            // for (var h = 0; h < height; h++)
            // {
            //     for (var w = 0; w < width; w++)
            //     {
            //         for (var c = 0; c < channels; c++)
            //         {
            //             tensorProxy.data[batch, h, w, c] = fillValue;
            //         }
            //     }
            // }

            // This assumes that batch is axis 0.
            var length = tensorProxy.data.shape.length;
            var batchsize = tensorProxy.data.shape[0];
            var start = batch * length / batchsize;
            var end = (batch + 1) * length / batchsize;
            for (var i = start; i < end; i++)
            {
                tensorProxy.FloatData[i] = fillValue;
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
            //@Barracude4Upgrade:
            if (tensorProxy.valueType != DataType.Float)
            {
                throw new NotImplementedException("Only float data types are currently supported");
            }

            if (tensorProxy.data == null)
            {
                throw new ArgumentNullException();
            }

            //@Barracude4Upgrade:
            for (var i = 0; i < tensorProxy.data.shape.length; i++)
            {
                tensorProxy.FloatData[i] = (float)randomNormal.NextDouble();
            }
        }
    }
}

