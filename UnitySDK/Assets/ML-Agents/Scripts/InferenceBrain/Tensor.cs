using System;
using System.Collections.Generic;
using Barracuda;

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
	public class Tensor
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
		//public Array Data;
		public Barracuda.Tensor Data;
	}
	
	public class TensorUtils
	{
		public static void ResizeTensor(Tensor tensor, int batch, ITensorAllocator allocator)
		{
			if (tensor.Shape[0] == batch &&
			    tensor.Data != null && tensor.Data.batch == batch)
				return; 

			tensor.Data?.Dispose();
			tensor.Shape[0] = batch;
			tensor.Data = allocator.Alloc(new TensorShape(batch, (int)tensor.Shape[tensor.Shape.Length - 1]));
		}
	}

}
