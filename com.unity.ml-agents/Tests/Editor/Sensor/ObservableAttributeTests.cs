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
        }

        [Test]
        public void TestGetObservableSensors()
        {
            var testClass = new TestClass();
            testClass.m_IntMember = 1;
            testClass.IntProperty = 2;

            var sensors = ObservableAttribute.GetObservableSensors(testClass);
            Assert.AreEqual(sensors.Count, 2);

            var sensorsByName = new Dictionary<string, ISensor>();
            foreach (var sensor in sensors)
            {
                sensorsByName[sensor.GetName()] = sensor;
            }

            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.m_IntMember"], new[] {1.0f});
            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.IntProperty"], new[] {2.0f});

        }
    }
}
