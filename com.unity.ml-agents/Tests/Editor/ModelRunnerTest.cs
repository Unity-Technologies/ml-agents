using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class ModelRunnerTest
    {
        const string k_continuous2vis8vec2actionPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action.nn";
        const string k_discrete1vis0vec_2_3action_recurrModelPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/discrete1vis0vec_2_3action_recurr.nn";
        NNModel continuous2vis8vec2actionModel;
        NNModel discrete1vis0vec_2_3action_recurrModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;

        ActionSpec GetContinuous2vis8vec2actionActionSpec()
        {
            return ActionSpec.MakeContinuous(2);
        }

        ActionSpec GetDiscrete1vis0vec_2_3action_recurrModelActionSpec()
        {
            return ActionSpec.MakeDiscrete(2, 3);
        }

        [SetUp]
        public void SetUp()
        {
            continuous2vis8vec2actionModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_continuous2vis8vec2actionPath, typeof(NNModel));
            discrete1vis0vec_2_3action_recurrModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_discrete1vis0vec_2_3action_recurrModelPath, typeof(NNModel));
            var go = new GameObject("SensorA");
            sensor_21_20_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_21_20_3.Sensor = new Test3DSensor("SensorA", 21, 20, 3);
            sensor_20_22_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_20_22_3.Sensor = new Test3DSensor("SensorB", 20, 22, 3);
        }

        [Test]
        public void TestModelExist()
        {
            Assert.IsNotNull(continuous2vis8vec2actionModel);
            Assert.IsNotNull(discrete1vis0vec_2_3action_recurrModel);
        }

        [Test]
        public void TestCreation()
        {
            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionActionSpec());
            modelRunner.Dispose();
            modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, GetDiscrete1vis0vec_2_3action_recurrModelActionSpec());
            modelRunner.Dispose();
        }

        [Test]
        public void TestHasModel()
        {
            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionActionSpec(), InferenceDevice.CPU);
            Assert.True(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.CPU));
            Assert.False(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.GPU));
            Assert.False(modelRunner.HasModel(discrete1vis0vec_2_3action_recurrModel, InferenceDevice.CPU));
            modelRunner.Dispose();
        }

        [Test]
        public void TestRunModel()
        {
            var actionSpec = GetDiscrete1vis0vec_2_3action_recurrModelActionSpec();
            var modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, actionSpec);
            var info1 = new AgentInfo();
            info1.episodeId = 1;
            modelRunner.PutObservations(info1, new[] { sensor_21_20_3.CreateSensor() }.ToList());
            var info2 = new AgentInfo();
            info2.episodeId = 2;
            modelRunner.PutObservations(info2, new[] { sensor_21_20_3.CreateSensor() }.ToList());

            modelRunner.DecideBatch();

            Assert.IsNotNull(modelRunner.GetAction(1));
            Assert.IsNotNull(modelRunner.GetAction(2));
            Assert.IsNull(modelRunner.GetAction(3));
            Assert.AreEqual(actionSpec.NumDiscreteActions, modelRunner.GetAction(1).Count());
            modelRunner.Dispose();
        }
    }
}
