using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Barracuda;
using UnityEngine;

namespace MLAgents.InferenceBrain
{

	/// <summary>
	/// Tensor - A class to encapsulate a Tensor used for inference.
	/// 
	/// This class contains the Array that holds the data array, the shapes, type and the placeholder in the
	/// execution graph. All the fields are editable in the inspector, allowing the user to specify everything
	/// but the data in a graphical way.
	/// </summary>
	[System.Serializable]
	public class TensorProxy
	{
		public enum TensorType
		{
			Integer,
			FloatingPoint
		};

		private static Dictionary<TensorType, Type> m_typeMap = new Dictionary<TensorType, Type>()
		{
			{ TensorType.FloatingPoint, typeof(float)},
			{TensorType.Integer, typeof(int)}
		};

		public string Name;
		public TensorType ValueType;
		// Since Type is not serializable, we use the DisplayType for the Inspector
		public Type DataType
		{
			get { return m_typeMap[ValueType]; }
		}
		public long[] Shape;
		
		public Tensor Data;
	}
	
	public class TensorUtils
	{
		public static void ResizeTensor(TensorProxy tensor, int batch, ITensorAllocator allocator)
		{
			if (tensor.Shape[0] == batch &&
			    tensor.Data != null && tensor.Data.batch == batch)
				return; 

			tensor.Data?.Dispose();
			tensor.Shape[0] = batch;
			
			if (tensor.Shape.Length == 4)
				tensor.Data = allocator.Alloc(new TensorShape(batch, (int)tensor.Shape[1], (int)tensor.Shape[2], (int)tensor.Shape[3]));
			else
				tensor.Data = allocator.Alloc(new TensorShape(batch, (int)tensor.Shape[tensor.Shape.Length - 1]));
		}

		public static Array BarracudaToFloatArray(Tensor tensor)
		{
			Array res;
			
			if (tensor.height == 1 && tensor.width == 1)
				res = new float[tensor.batch, tensor.channels];
			else
				res = new float[tensor.batch, tensor.height, tensor.width, tensor.channels];
			
			Buffer.BlockCopy(tensor.readonlyArray, 0, res, 0, tensor.length * Marshal.SizeOf<float>());

			return res;
		}
		
		public static Array BarracudaToIntArray(Tensor tensor)
		{

			if (tensor.height == 1 && tensor.width == 1)
			{
				var res = new int[tensor.batch, tensor.channels];
				
				for (int b = 0; b < tensor.batch; b++)
				for (int c = 0; c < tensor.channels; c++)
				{
					res[b, c] = (int)tensor[b, c];
				}

				return res;
			}
			else
			{
				var res = new int[tensor.batch, tensor.height, tensor.width, tensor.channels];
				for (int b = 0; b < tensor.batch; b++)
				for (int y = 0; y < tensor.height; y++)
				for (int x = 0; x < tensor.width; x++)
				for (int c = 0; c < tensor.channels; c++)
				{
					res[b, y, x, c] = (int)tensor[b, y, x, c];
				}

				return res;
			}
		}

		public static Tensor ArrayToBarracuda(Array array)
		{
			Tensor res;
			
			if (array.Rank == 2)
				res = new Tensor(array.GetLength(0), array.GetLength(1));
			else
				res = new Tensor(array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3));

			int offset = 0;
			var barracudaArray = res.data != null ? res.tensorOnDevice.SharedAccess(out offset) : null;

			Buffer.BlockCopy(array, 0, barracudaArray, offset, res.length * Marshal.SizeOf<float>());
			
			return res;
		}

		internal static long[] TensorShapeFromBarracuda(TensorShape src)
		{
			if (src.height == 1 && src.width == 1)
				return new long[2] {src.batch, src.channels};

			return new long[4] {src.batch, src.height, src.width, src.channels};
		}

		public static TensorProxy TensorProxyFromBarracuda(Tensor src, string nameOverride = null)
		{
			var shape = TensorShapeFromBarracuda(src.shape);
			return new TensorProxy
			{
				Name = nameOverride ?? src.name,
				ValueType = TensorProxy.TensorType.FloatingPoint,
				Shape = shape,
				Data = src
			};
		}
	}

}
