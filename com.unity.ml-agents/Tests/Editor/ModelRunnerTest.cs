using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Barracuda;
using MLAgents.Inference;
using MLAgents.Sensors;
using System.Linq;
using MLAgents.Policies;

namespace MLAgents.Tests
{
    [TestFixture]
    public class ModelRunnerTest
    {
        const string k_continuous2vis8vec2actionPath = "Packages/com.unity.ml-agents/Tests/Editor/Resources/continuous2vis8vec2action.nn";
        const string k_discrete1vis0vec_2_3action_recurrModelPath = "Packages/com.unity.ml-agents/Tests/Editor/Resources/discrete1vis0vec_2_3action_recurr.nn";
        NNModel continuous2vis8vec2actionModel;
        NNModel discrete1vis0vec_2_3action_recurrModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;

        private BrainParameters GetContinuous2vis8vec2actionBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.vectorObservationSize = 8;
            validBrainParameters.vectorActionSize = new int[] { 2 };
            validBrainParameters.numStackedVectorObservations = 1;
            validBrainParameters.vectorActionSpaceType = SpaceType.Continuous;
            return validBrainParameters;
        }

        private BrainParameters GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.vectorObservationSize = 0;
            validBrainParameters.vectorActionSize = new int[] { 2, 3 };
            validBrainParameters.numStackedVectorObservations = 1;
            validBrainParameters.vectorActionSpaceType = SpaceType.Discrete;
            return validBrainParameters;
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
            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionBrainParameters());
            modelRunner.Dispose();
            modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters());
            modelRunner.Dispose();
        }

        [Test]
        public void TestHasModel()
        {
            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionBrainParameters(), InferenceDevice.CPU);
            Assert.True(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.CPU));
            Assert.False(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.GPU));
            Assert.False(modelRunner.HasModel(discrete1vis0vec_2_3action_recurrModel, InferenceDevice.CPU));
            modelRunner.Dispose();
        }

        [Test]
        public void TestRunModel()
        {
            var brainParameters = GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters();
            var modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, brainParameters);
            var info1 = new AgentInfo();
            info1.episodeId = 1;
            modelRunner.PutObservations(info1, new ISensor[] { sensor_21_20_3.CreateSensor() }.ToList());
            var info2 = new AgentInfo();
            info2.episodeId = 2;
            modelRunner.PutObservations(info2, new ISensor[] { sensor_21_20_3.CreateSensor() }.ToList());

            modelRunner.DecideBatch();

            Assert.IsNotNull(modelRunner.GetAction(1));
            Assert.IsNotNull(modelRunner.GetAction(2));
            Assert.IsNull(modelRunner.GetAction(3));
            Assert.AreEqual(brainParameters.vectorActionSize.Count(), modelRunner.GetAction(1).Count());
            modelRunner.Dispose();
        }
    }
}
