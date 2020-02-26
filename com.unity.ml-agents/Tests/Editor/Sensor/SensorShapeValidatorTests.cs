using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace MLAgents.Tests
{
    public class NoopSensor : ISensor
    {
        string m_Name = "NoopSensor";
        int[] m_Shape;

        public NoopSensor(int dim1)
        {
            m_Shape = new[] { dim1 };
        }

        public NoopSensor(int dim1, int dim2)
        {
            m_Shape = new[] { dim1, dim2, };
        }

        public NoopSensor(int dim1, int dim2, int dim3)
        {
            m_Shape = new[] { dim1, dim2, dim3};
        }

        public string GetName()
        {
            return m_Name;
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public int Write(WriteAdapter adapter)
        {
            return this.ObservationSize();
        }

        public void Update() { }

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }
    }

    public class SensorShapeValidatorTests
    {
        [Test]
        public void TestSizeMismatch()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new NoopSensor(1), new NoopSensor(2, 3), new NoopSensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new NoopSensor(1), new NoopSensor(2, 3), new NoopSensor(4, 5, 7) };
            LogAssert.Expect(LogType.Assert, "Sensor sizes much match.");
            validator.ValidateSensors(sensorList2);

        }
    }
}
