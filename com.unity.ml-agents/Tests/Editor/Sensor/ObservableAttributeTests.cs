using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;

namespace Unity.MLAgents.Tests
{

    [TestFixture]
    public class ObservableAttributeTests
    {
        class TestClass
        {
            [Observable]
            public int m_IntMember;

            int m_IntProperty;

            [Observable]
            public int IntProperty
            {
                get => m_IntProperty;
                set => m_IntProperty = value;
            }

            [Observable("vector3member")]
            public Vector3 m_Vector3Member;

            Vector3 m_VectorProperty;

            [Observable("vector3property")]
            public Vector3 VectorProperty
            {
                get => m_VectorProperty;
                set => m_VectorProperty = value;
            }
        }

        [Test]
        public void TestGetObservableSensors()
        {
            var testClass = new TestClass();
            testClass.m_IntMember = 1;
            testClass.IntProperty = 2;
            testClass.m_Vector3Member = new Vector3(30,31,32);
            testClass.VectorProperty = new Vector3(33,34,35);

            var sensors = ObservableAttribute.GetObservableSensors(testClass);
            Assert.AreEqual(sensors.Count, 4);

            var sensorsByName = new Dictionary<string, ISensor>();
            foreach (var sensor in sensors)
            {
                sensorsByName[sensor.GetName()] = sensor;
            }

            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.m_IntMember"], new[] {1.0f});
            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.IntProperty"], new[] {2.0f});
            SensorTestHelper.CompareObservation(sensorsByName["vector3member"], new[] {30.0f, 31.0f, 32.0f});
            SensorTestHelper.CompareObservation(sensorsByName["vector3property"], new[] {33.0f, 34.0f, 35.0f});

        }
    }
}
