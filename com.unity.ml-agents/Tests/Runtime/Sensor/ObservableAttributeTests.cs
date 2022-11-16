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
        public enum TestEnum
        {
            ValueA = -100,
            ValueB = 1,
            ValueC = 42,
        }

        [Flags]
        public enum TestFlags
        {
            FlagA = 1,
            FlagB = 2,
            FlagC = 4
        }

        class TestClass
        {
            // Non-observables
            int m_NonObservableInt;
            float m_NonObservableFloat;

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

            //
            // Enum
            //

            [Observable("enumMember")]
            public TestEnum m_EnumMember = TestEnum.ValueA;

            TestEnum m_EnumProperty = TestEnum.ValueC;

            [Observable("enumProperty")]
            public TestEnum EnumProperty
            {
                get => m_EnumProperty;
                set => m_EnumProperty = value;
            }

            [Observable("badEnumMember")]
            public TestEnum m_BadEnumMember = (TestEnum)1337;

            //
            // Flags
            //
            [Observable("flagMember")]
            public TestFlags m_FlagMember = TestFlags.FlagA;

            TestFlags m_FlagProperty = TestFlags.FlagB | TestFlags.FlagC;

            [Observable("flagProperty")]
            public TestFlags FlagProperty
            {
                get => m_FlagProperty;
                set => m_FlagProperty = value;
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

            var sensors = ObservableAttribute.CreateObservableSensors(testClass, false);

            var sensorsByName = new Dictionary<string, ISensor>();
            foreach (var sensor in sensors)
            {
                sensorsByName[sensor.GetName()] = sensor;
            }

            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.m_IntMember"], new[] { 1.0f });
            SensorTestHelper.CompareObservation(sensorsByName["ObservableAttribute:TestClass.IntProperty"], new[] { 2.0f });

            SensorTestHelper.CompareObservation(sensorsByName["floatMember"], new[] { 1.1f });
            SensorTestHelper.CompareObservation(sensorsByName["floatProperty"], new[] { 1.2f });

            SensorTestHelper.CompareObservation(sensorsByName["boolMember"], new[] { 1.0f });
            SensorTestHelper.CompareObservation(sensorsByName["boolProperty"], new[] { 1.0f });

            SensorTestHelper.CompareObservation(sensorsByName["vector2Member"], new[] { 2.0f, 2.1f });
            SensorTestHelper.CompareObservation(sensorsByName["vector2Property"], new[] { 2.2f, 2.3f });

            SensorTestHelper.CompareObservation(sensorsByName["vector3Member"], new[] { 3.0f, 3.1f, 3.2f });
            SensorTestHelper.CompareObservation(sensorsByName["vector3Property"], new[] { 3.3f, 3.4f, 3.5f });

            SensorTestHelper.CompareObservation(sensorsByName["vector4Member"], new[] { 4.0f, 4.1f, 4.2f, 4.3f });
            SensorTestHelper.CompareObservation(sensorsByName["vector4Property"], new[] { 4.4f, 4.5f, 4.5f, 4.7f });

            SensorTestHelper.CompareObservation(sensorsByName["quaternionMember"], new[] { 5.0f, 5.1f, 5.2f, 5.3f });
            SensorTestHelper.CompareObservation(sensorsByName["quaternionProperty"], new[] { 5.4f, 5.5f, 5.5f, 5.7f });

            // Actual ordering is B, C, A
            SensorTestHelper.CompareObservation(sensorsByName["enumMember"], new[] { 0.0f, 0.0f, 1.0f });
            SensorTestHelper.CompareObservation(sensorsByName["enumProperty"], new[] { 0.0f, 1.0f, 0.0f });
            SensorTestHelper.CompareObservation(sensorsByName["badEnumMember"], new[] { 0.0f, 0.0f, 0.0f });

            SensorTestHelper.CompareObservation(sensorsByName["flagMember"], new[] { 1.0f, 0.0f, 0.0f });
            SensorTestHelper.CompareObservation(sensorsByName["flagProperty"], new[] { 0.0f, 1.0f, 1.0f });
        }

        [Test]
        public void TestGetTotalObservationSize()
        {
            var testClass = new TestClass();
            var errors = new List<string>();
            var expectedObsSize = 2 * ( // two fields each of these
                    1 // int
                    + 1 // float
                    + 1 // bool
                    + 2 // vector2
                    + 3 // vector3
                    + 4 // vector4
                    + 4 // quaternion
                    + 3 // TestEnum - 3 values
                    + 3 // TestFlags - 3 values
                )
                + 3; // TestEnum with bad value
            Assert.AreEqual(expectedObsSize, ObservableAttribute.GetTotalObservationSize(testClass, false, errors));
            Assert.AreEqual(0, errors.Count);
        }

        class BadClass
        {
            [Observable]
            double m_Double;

            [Observable]
            double DoubleProperty
            {
                get => m_Double;
                set => m_Double = value;
            }

            float m_WriteOnlyProperty;

            [Observable]
            // No get property, so we shouldn't be able to make a sensor out of this.
            public float WriteOnlyProperty
            {
                set => m_WriteOnlyProperty = value;
            }
        }

        [Test]
        public void TestInvalidObservables()
        {
            var bad = new BadClass();
            bad.WriteOnlyProperty = 1.0f;
            var errors = new List<string>();
            Assert.AreEqual(0, ObservableAttribute.GetTotalObservationSize(bad, false, errors));
            Assert.AreEqual(3, errors.Count);

            // Should be able to safely generate sensors (and get nothing back)
            var sensors = ObservableAttribute.CreateObservableSensors(bad, false);
            Assert.AreEqual(0, sensors.Count);
        }

        class StackingClass
        {
            [Observable(numStackedObservations: 2)]
            public float FloatVal;
        }

        [Test]
        public void TestObservableAttributeStacking()
        {
            var c = new StackingClass();
            c.FloatVal = 1.0f;
            var sensors = ObservableAttribute.CreateObservableSensors(c, false);
            var sensor = sensors[0];
            Assert.AreEqual(typeof(StackingSensor), sensor.GetType());
            SensorTestHelper.CompareObservation(sensor, new[] { 0.0f, 1.0f });

            sensor.Update();
            c.FloatVal = 3.0f;
            SensorTestHelper.CompareObservation(sensor, new[] { 1.0f, 3.0f });

            var errors = new List<string>();
            Assert.AreEqual(2, ObservableAttribute.GetTotalObservationSize(c, false, errors));
            Assert.AreEqual(0, errors.Count);
        }

        class BaseClass
        {
            [Observable("base")]
            public float m_BaseField;

            [Observable("private")]
            float m_PrivateField;
        }

        class DerivedClass : BaseClass
        {
            [Observable("derived")]
            float m_DerivedField;
        }

        [Test]
        public void TestObservableAttributeExcludeInherited()
        {
            var d = new DerivedClass();
            d.m_BaseField = 1.0f;

            // excludeInherited=false will get fields in the derived class, plus public and protected inherited fields
            var sensorAll = ObservableAttribute.CreateObservableSensors(d, false);
            Assert.AreEqual(2, sensorAll.Count);
            // Note - actual order doesn't matter here, we can change this to use a HashSet if neeed.
            Assert.AreEqual("derived", sensorAll[0].GetName());
            Assert.AreEqual("base", sensorAll[1].GetName());

            // excludeInherited=true will only get fields in the derived class
            var sensorsDerivedOnly = ObservableAttribute.CreateObservableSensors(d, true);
            Assert.AreEqual(1, sensorsDerivedOnly.Count);
            Assert.AreEqual("derived", sensorsDerivedOnly[0].GetName());

            var b = new BaseClass();
            var baseSensors = ObservableAttribute.CreateObservableSensors(b, false);
            Assert.AreEqual(2, baseSensors.Count);
        }
    }
}
