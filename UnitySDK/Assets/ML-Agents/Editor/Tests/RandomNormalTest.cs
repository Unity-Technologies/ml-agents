using System;
using NUnit.Framework;
using MLAgents.InferenceBrain;
using MLAgents.InferenceBrain.Utils;

namespace MLAgents.Tests
{

	public class RandomNormalTest
	{

		[Test]
		public void RandomNormalTestTwoDouble()
		{
			RandomNormal rn = new RandomNormal(2018);

			Assert.AreEqual(-0.46666, rn.NextDouble(), 0.0001);
			Assert.AreEqual(-0.37989, rn.NextDouble(), 0.0001);
		}

		[Test]
		public void RandomNormalTestWithMean()
		{
			RandomNormal rn = new RandomNormal(2018, 5.0f);

			Assert.AreEqual(4.53333, rn.NextDouble(), 0.0001);
			Assert.AreEqual(4.6201, rn.NextDouble(), 0.0001);
		}

		[Test]
		public void RandomNormalTestWithStddev()
		{
			RandomNormal rn = new RandomNormal(2018, 1.0f, 4.2f);

			Assert.AreEqual(-0.9599, rn.NextDouble(), 0.0001);
			Assert.AreEqual(-0.5955, rn.NextDouble(), 0.0001);
		}

		[Test]
		public void RandomNormalTestWithMeanStddev()
		{
			RandomNormal rn = new RandomNormal(2018, -3.2f, 2.2f);

			Assert.AreEqual(-4.2266, rn.NextDouble(), 0.0001);
			Assert.AreEqual(-4.0357, rn.NextDouble(), 0.0001);
		}

		[Test]
		public void RandomNormalTestTensorInt()
		{
			RandomNormal rn = new RandomNormal(1982);
			Tensor t = new Tensor
			{
				ValueType = Tensor.TensorType.Integer
			};

			Assert.Throws<NotImplementedException>(() => rn.FillTensor(t));
		}

		[Test]
		public void RandomNormalTestDataNull()
		{
			RandomNormal rn = new RandomNormal(1982);
			Tensor t = new Tensor
			{
				ValueType = Tensor.TensorType.FloatingPoint
			};

			Assert.Throws<ArgumentNullException>(() => rn.FillTensor(t));
		}

		[Test]
		public void RandomNormalTestTensor()
		{
			RandomNormal rn = new RandomNormal(1982);
			Tensor t = new Tensor
			{
				ValueType = Tensor.TensorType.FloatingPoint,
				Data = Array.CreateInstance(typeof(float), new long[3] {3, 4, 2})
		};

			rn.FillTensor(t);

			float[] reference = new float[]
			{
				-0.2139822f,
				0.5051259f,
				-0.5640336f,
				-0.3357787f,
				-0.2055894f,
				-0.09432302f,
				-0.01419199f,
				0.53621f,
				-0.5507085f,
				-0.2651141f,
				0.09315512f,
				-0.04918706f,
				-0.179625f,
				0.2280539f,
				0.1883962f,
				0.4047216f,
				0.1704049f,
				0.5050544f,
				-0.3365685f,
				0.3542781f,
				0.5951571f,
				0.03460682f,
				-0.5537263f,
				-0.4378373f,
			};

			int i = 0;
			foreach (float f in t.Data)
			{
				Assert.AreEqual(f, reference[i], 0.0001);
				++i;
			}


		}
	}
}
