using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Unity.Sentis;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Analytics;
using UnityEditor;


namespace Unity.MLAgents.Tests.Analytics
{
    [TestFixture]
    public class InferenceAnalyticsTests
    {
        const string k_continuousONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action_v1_0.onnx";
        ModelAsset continuousONNXModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;

        ActionSpec GetContinuous2vis8vec2actionActionSpec()
        {
            return ActionSpec.MakeContinuous(2);
        }

        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }

            continuousONNXModel = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_continuousONNXPath, typeof(ModelAsset));
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
            var actionSpec = GetContinuous2vis8vec2actionActionSpec();

            var vectorActuator = new VectorActuator(null, actionSpec, "test'");
            var actuators = new IActuator[] { vectorActuator };

            var continuousEvent = InferenceAnalytics.GetEventForModel(
                continuousONNXModel, behaviorName,
                InferenceDevice.Burst, sensors, actionSpec,
                actuators
            );

            // The behavior name should be hashed, not pass-through.
            Assert.AreNotEqual(behaviorName, continuousEvent.BehaviorName);

            Assert.AreEqual(2, continuousEvent.ActionSpec.NumContinuousActions);
            Assert.AreEqual(0, continuousEvent.ActionSpec.NumDiscreteActions);
            Assert.AreEqual(2, continuousEvent.ObservationSpecs.Count);
            Assert.AreEqual(3, continuousEvent.ObservationSpecs[0].DimensionInfos.Length);
            Assert.AreEqual(20, continuousEvent.ObservationSpecs[0].DimensionInfos[1].Size);
            Assert.AreEqual(0, continuousEvent.ObservationSpecs[0].ObservationType);
            Assert.AreEqual((int)DimensionProperty.TranslationalEquivariance, continuousEvent.ObservationSpecs[0].DimensionInfos[1].Flags);
            Assert.AreEqual((int)DimensionProperty.None, continuousEvent.ObservationSpecs[0].DimensionInfos[0].Flags);
            Assert.AreEqual("None", continuousEvent.ObservationSpecs[0].CompressionType);
            Assert.AreEqual(Test3DSensor.k_BuiltInSensorType, continuousEvent.ObservationSpecs[0].BuiltInSensorType);
            Assert.AreEqual((int)BuiltInActuatorType.VectorActuator, continuousEvent.ActuatorInfos[0].BuiltInActuatorType);
            Assert.AreNotEqual(null, continuousEvent.ModelHash);

            // Make sure nested fields get serialized
            var jsonString = JsonUtility.ToJson(continuousEvent, true);
            Assert.IsTrue(jsonString.Contains("ObservationSpecs"));
            Assert.IsTrue(jsonString.Contains("ActionSpec"));
            Assert.IsTrue(jsonString.Contains("NumDiscreteActions"));
            Assert.IsTrue(jsonString.Contains("SensorName"));
            Assert.IsTrue(jsonString.Contains("Flags"));
            Assert.IsTrue(jsonString.Contains("ActuatorInfos"));
        }

        [Test]
        public void TestSentisPolicy()
        {
            // Explicitly request decisions for a policy so we get code coverage on the event sending
            using (new AnalyticsUtils.DisableAnalyticsSending())
            {
                var sensors = new List<ISensor> { sensor_21_20_3.Sensor, sensor_20_22_3.Sensor };
                var policy = new SentisPolicy(
                    GetContinuous2vis8vec2actionActionSpec(),
                    Array.Empty<IActuator>(),
                    continuousONNXModel,
                    InferenceDevice.Burst,
                    "testBehavior"
                );
                policy.RequestDecision(new AgentInfo(), sensors);
            }
            Academy.Instance.Dispose();
        }
    }
}
