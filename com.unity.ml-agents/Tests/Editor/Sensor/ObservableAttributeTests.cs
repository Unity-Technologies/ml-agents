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
            //
            // Int
            //
            [Observable]
            public int m_IntMember;

            int m_IntProperty;

            [Observable]
            public int IntProperty
            {
                get => m_IntProperty;
                set => m_IntProperty = value;
            }

            //
            // Float
            //
            [Observable("floatMember")]
            public float m_FloatMember;

            float m_FloatProperty;
            [Observable("floatProperty")]
            public float FloatProperty
            {
                get => m_FloatProperty;
                set => m_FloatProperty = value;
            }

            //
            // Bool
            //
            [Observable("boolMember")]
            public bool m_BoolMember;

            bool m_BoolProperty;
            [Observable("boolProperty")]
            public bool BoolProperty
            {
                get => m_BoolProperty;
                set => m_BoolProperty = value;
            }

            //
            // Vector2
            //

            [Observable("vector2Member")]
            public Vector2 m_Vector2Member;

            Vector2 m_Vector2Property;

            [Observable("vector2Property")]
            public Vector2 Vector2Property
            {
                get => m_Vector2Property;
                set => m_Vector2Property = value;
            }

            //
            // Vector3
            //
            [Observable("vector3Member")]
            public Vector3 m_Vector3Member;

            Vector3 m_Vector3Property;

            [Observable("vector3Property")]
            public Vector3 Vector3Property
            {
                get => m_Vector3Property;
                set => m_Vector3Property = value;
            }

            //
            // Vector4
            //

            [Observable("vector4Member")]
            public Vector4 m_Vector4Member;

            Vector4 m_Vector4Property;

            [Observable("vector4Property")]
            public Vector4 Vector4Property
            {
                get => m_Vector4Property;
                set => m_Vector4Property = value;
            }

            //
            // Quaternion
            //
            [Observable("quaternionMember")]
            public Quaternion m_QuaternionMember;

            Quaternion m_QuaternionProperty;

            [Observable("quaternionProperty")]
            public Quaternion QuaternionProperty
            {
                get => m_QuaternionProperty;
                set => m_QuaternionProperty = value;
            }
        }

        [Test]
        public void TestGetObservableSensors()
        {
            var testClass = new TestClass();
            testClass.m_IntMember = 1;
            testClass.IntProperty = 2;

            testClass.m_FloatMember = 1.1f;
            testClass.FloatProperty = 1.2f;

            testClass.m_BoolMember = true;
            testClass.BoolProperty = true;

            testClass.m_Vector2Member = new Vector2(2.0f, 2.1f);
            testClass.Vector2Property = new Vector2(2.2f, 2.3f);

            testClass.m_Vector3Member = new Vector3(3.0f, 3.1f, 3.2f);
            testClass.Vector3Property = new Vector3(3.3f, 3.4f, 3.5f);

            testClass.m_Vector4Member = new Vector4(4.0f, 4.1f, 4.2f, 4.3f);
            testClass.Vector4Property = new Vector4(4.4f, 4.5f, 4.5f, 4.7f);

            testClass.m_Vector4Member = new Vector4(4.0f, 4.1f, 4.2f, 4.3f);
            testClass.Vector4Property = new Vector4(4.4f, 4.5f, 4.5f, 4.7f);

            testClass.m_QuaternionMember = new Quaternion(5.0f, 5.1f, 5.2f, 5.3f);
            testClass.QuaternionProperty = new Quaternion(5.4f, 5.5f, 5.5f, 5.7f);

            var sensors = ObservableAttribute.GetObservableSensors(testClass);

            var sensorsByName = new Dictionary<string, ISensor>();
            foreach (var sensor in sensors)
            {
                sensorsByName[sensor.GetName()] = sensor;
            }

            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.m_IntMember"], new[] {1.0f});
            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.IntProperty"], new[] {2.0f});

            SensorTestHelper.CompareObservation(sensorsByName["floatMember"], new[] {1.1f});
            SensorTestHelper.CompareObservation(sensorsByName["floatProperty"], new[] {1.2f});

            SensorTestHelper.CompareObservation(sensorsByName["boolMember"], new[] {1.0f});
            SensorTestHelper.CompareObservation(sensorsByName["boolProperty"], new[] {1.0f});

            SensorTestHelper.CompareObservation(sensorsByName["vector2Member"], new[] {2.0f, 2.1f});
            SensorTestHelper.CompareObservation(sensorsByName["vector2Property"], new[] {2.2f, 2.3f});

            SensorTestHelper.CompareObservation(sensorsByName["vector3Member"], new[] {3.0f, 3.1f, 3.2f});
            SensorTestHelper.CompareObservation(sensorsByName["vector3Property"], new[] {3.3f, 3.4f, 3.5f});

            SensorTestHelper.CompareObservation(sensorsByName["vector4Member"], new[] {4.0f, 4.1f, 4.2f, 4.3f});
            SensorTestHelper.CompareObservation(sensorsByName["vector4Property"], new[] {4.4f, 4.5f, 4.5f, 4.7f});

            SensorTestHelper.CompareObservation(sensorsByName["quaternionMember"], new[] {5.0f, 5.1f, 5.2f, 5.3f});
            SensorTestHelper.CompareObservation(sensorsByName["quaternionProperty"], new[] {5.4f, 5.5f, 5.5f, 5.7f});
        }
    }
}
