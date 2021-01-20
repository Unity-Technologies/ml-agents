using System;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{

    [TestFixture]
    public class BufferSensorTest
    {
        [Test]
        public void TestBufferSensor()
        {

            var bufferSensor = new BufferSensor(20, 4);
            var shape = bufferSensor.GetObservationShape();
            var dimProp = bufferSensor.GetDimensionProperties();
            Assert.AreEqual(shape[0], 20);
            Assert.AreEqual(shape[1], 4);
            Assert.AreEqual(shape.Length, 2);
            Assert.AreEqual(dimProp[0], DimensionProperty.VariableSize);
            Assert.AreEqual(dimProp[1], DimensionProperty.None);
            Assert.AreEqual(dimProp.Length, 2);

            bufferSensor.AppendObservation(new float[] { 1, 2, 3, 4 });
            bufferSensor.AppendObservation(new float[] { 5, 6, 7, 8 });

            var obsWriter = new ObservationWriter();
            var obs = bufferSensor.GetObservationProto(obsWriter);

            Assert.AreEqual(shape, obs.Shape);
            Assert.AreEqual(obs.DimensionProperties.Count, 2);
            Assert.AreEqual((int)dimProp[0], obs.DimensionProperties[0]);
            Assert.AreEqual((int)dimProp[1], obs.DimensionProperties[1]);

            for (int i = 0; i < 8; i++)
            {
                Assert.AreEqual(obs.FloatData.Data[i], i + 1);

            }
            for (int i = 8; i < 80; i++)
            {
                Assert.AreEqual(obs.FloatData.Data[i], 0);

            }
        }

        [Test]
        public void TestBufferSensorComponent()
        {
            var agentGameObj = new GameObject("agent");
            var bufferComponent = agentGameObj.AddComponent<BufferSensorComponent>();
            bufferComponent.MaxNumObservables = 20;
            bufferComponent.ObservableSize = 4;

            var sensor = bufferComponent.CreateSensor();
            var shape = bufferComponent.GetObservationShape();

            Assert.AreEqual(shape[0], 20);
            Assert.AreEqual(shape[1], 4);
            Assert.AreEqual(shape.Length, 2);

            bufferComponent.AppendObservation(new float[] { 1, 2, 3, 4 });
            bufferComponent.AppendObservation(new float[] { 5, 6, 7, 8 });

            var obsWriter = new ObservationWriter();
            var obs = sensor.GetObservationProto(obsWriter);

            Assert.AreEqual(shape, obs.Shape);
            Assert.AreEqual(obs.DimensionProperties.Count, 2);

            for (int i = 0; i < 8; i++)
            {
                Assert.AreEqual(obs.FloatData.Data[i], i + 1);

            }
            for (int i = 8; i < 80; i++)
            {
                Assert.AreEqual(obs.FloatData.Data[i], 0);

            }
        }

    }
}
