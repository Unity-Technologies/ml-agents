using NUnit.Framework;
using Unity.MLAgents.Sensors;


namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class ObservationSpecTests
    {
        [Test]
        public void TestVectorObsSpec()
        {
            var obsSpec = ObservationSpec.Vector(5);
            Assert.AreEqual(1, obsSpec.Rank);

            var shape = obsSpec.Shape;
            Assert.AreEqual(1, shape.Length);
            Assert.AreEqual(5, shape[0]);

            var dimensionProps = obsSpec.DimensionProperties;
            Assert.AreEqual(1, dimensionProps.Length);
            Assert.AreEqual(DimensionProperty.None, dimensionProps[0]);

            Assert.AreEqual(ObservationType.Default, obsSpec.ObservationType);
        }

        [Test]
        public void TestVariableLengthObsSpec()
        {
            var obsSpec = ObservationSpec.VariableLength(5, 6);
            Assert.AreEqual(2, obsSpec.Rank);

            var shape = obsSpec.Shape;
            Assert.AreEqual(2, shape.Length);
            Assert.AreEqual(5, shape[0]);
            Assert.AreEqual(6, shape[1]);

            var dimensionProps = obsSpec.DimensionProperties;
            Assert.AreEqual(2, dimensionProps.Length);
            Assert.AreEqual(DimensionProperty.VariableSize, dimensionProps[0]);
            Assert.AreEqual(DimensionProperty.None, dimensionProps[1]);

            Assert.AreEqual(ObservationType.Default, obsSpec.ObservationType);
        }

        [Test]
        public void TestVisualObsSpec()
        {
            var obsSpec = ObservationSpec.Visual(5, 6, 7);
            Assert.AreEqual(3, obsSpec.Rank);

            var shape = obsSpec.Shape;
            Assert.AreEqual(3, shape.Length);
            Assert.AreEqual(5, shape[0]);
            Assert.AreEqual(6, shape[1]);
            Assert.AreEqual(7, shape[2]);

            var dimensionProps = obsSpec.DimensionProperties;
            Assert.AreEqual(3, dimensionProps.Length);
            Assert.AreEqual(DimensionProperty.TranslationalEquivariance, dimensionProps[1]);
            Assert.AreEqual(DimensionProperty.TranslationalEquivariance, dimensionProps[2]);
            Assert.AreEqual(DimensionProperty.None, dimensionProps[0]);

            Assert.AreEqual(ObservationType.Default, obsSpec.ObservationType);
        }

        [Test]
        public void TestMismatchShapeDimensionPropThrows()
        {
            var shape = new InplaceArray<int>(1, 2);
            var dimProps = new InplaceArray<DimensionProperty>(DimensionProperty.TranslationalEquivariance);
            Assert.Throws<UnityAgentsException>(() => new ObservationSpec(shape, dimProps));
        }
    }
}
