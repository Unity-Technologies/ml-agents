using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.Policies;
using UnityEditor;

namespace Unity.MLAgents.Tests.Analytics
{
    [TestFixture]
    public class InferenceAnalyticsTests
    {
        const string k_continuousONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action.onnx";
        NNModel continuousONNXModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;

        ActionSpec GetContinuous2vis8vec2actionActionSpec()
        {
            return ActionSpec.MakeContinuous(2);
        }

        [SetUp]
        public void SetUp()
        {
            continuousONNXModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_continuousONNXPath, typeof(NNModel));
            var go = new GameObject("SensorA");
            sensor_21_20_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_21_20_3.Sensor = new Test3DSensor("SensorA", 21, 20, 3);
            sensor_20_22_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_20_22_3.Sensor = new Test3DSensor("SensorB", 20, 22, 3);
        }

        [Test]
        public void TestModelEvent()
        {
            var sensors = new List<ISensor> { sensor_21_20_3.Sensor, sensor_20_22_3.Sensor };
            var behaviorName = "continuousModel";

            var continuousEvent = InferenceAnalytics.GetEventForModel(
                continuousONNXModel, behaviorName,
                InferenceDevice.CPU, sensors, GetContinuous2vis8vec2actionActionSpec()
            );

            // The behavior name should be hashed, not pass-through.
            Assert.AreNotEqual(behaviorName, continuousEvent.BehaviorName);

            Assert.AreEqual(2, continuousEvent.ActionSpec.NumContinuousActions);
            Assert.AreEqual(0, continuousEvent.ActionSpec.NumDiscreteActions);
            Assert.AreEqual(2, continuousEvent.ObservationSpecs.Count);
            Assert.AreEqual(3, continuousEvent.ObservationSpecs[0].DimensionInfos.Length);
            Assert.AreEqual(20, continuousEvent.ObservationSpecs[0].DimensionInfos[0].Size);
            Assert.AreEqual("None", continuousEvent.ObservationSpecs[0].CompressionType);
            Assert.AreNotEqual(null, continuousEvent.ModelHash);

            // Make sure nested fields get serialized
            var jsonString = JsonUtility.ToJson(continuousEvent, true);
            Assert.IsTrue(jsonString.Contains("ObservationSpecs"));
            Assert.IsTrue(jsonString.Contains("ActionSpec"));
            Assert.IsTrue(jsonString.Contains("NumDiscreteActions"));
            Assert.IsTrue(jsonString.Contains("SensorName"));
            Assert.IsTrue(jsonString.Contains("Flags"));
        }
    }
}
