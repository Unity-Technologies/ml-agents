using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Unity.Barracuda;
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

        BrainParameters GetContinuous2vis8vec2actionBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 8;
            validBrainParameters.VectorActionSize = new [] { 2 };
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.VectorActionSpaceType = SpaceType.Continuous;
            return validBrainParameters;
        }

        BrainParameters GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 0;
            validBrainParameters.VectorActionSize = new [] { 2, 3 };
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.VectorActionSpaceType = SpaceType.Discrete;
            return validBrainParameters;
        }

        //[SetUp]
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

//        [Test]
//        public void TestModelExist()
//        {
//            Assert.IsNotNull(continuous2vis8vec2actionModel);
//            Assert.IsNotNull(discrete1vis0vec_2_3action_recurrModel);
//        }
//
//        [Test]
//        public void TestCreation()
//        {
//            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionBrainParameters());
//            modelRunner.Dispose();
//            modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters());
//            modelRunner.Dispose();
//        }
//
//        [Test]
//        public void TestHasModel()
//        {
//            var modelRunner = new ModelRunner(continuous2vis8vec2actionModel, GetContinuous2vis8vec2actionBrainParameters(), InferenceDevice.CPU);
//            Assert.True(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.CPU));
//            Assert.False(modelRunner.HasModel(continuous2vis8vec2actionModel, InferenceDevice.GPU));
//            Assert.False(modelRunner.HasModel(discrete1vis0vec_2_3action_recurrModel, InferenceDevice.CPU));
//            modelRunner.Dispose();
//        }

        //[Test]
        public void TestRunModel()
        {
            var brainParameters = GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters();
            var modelRunner = new ModelRunner(discrete1vis0vec_2_3action_recurrModel, brainParameters);
            var info1 = new AgentInfo();
            info1.episodeId = 1;
            modelRunner.PutObservations(info1, new [] { sensor_21_20_3.CreateSensor() }.ToList());
            var info2 = new AgentInfo();
            info2.episodeId = 2;
            modelRunner.PutObservations(info2, new [] { sensor_21_20_3.CreateSensor() }.ToList());

            modelRunner.DecideBatch();

            Assert.IsNotNull(modelRunner.GetAction(1));
            Assert.IsNotNull(modelRunner.GetAction(2));
            Assert.IsNull(modelRunner.GetAction(3));
            Assert.AreEqual(brainParameters.VectorActionSize.Count(), modelRunner.GetAction(1).Count());
            modelRunner.Dispose();
        }
    }

    [TestFixture]
    public class BarracudaReproTest
    {
        private IWorker m_Worker;

        [SetUp]
        public void Setup()
        {
            var model = new Model();
            var builder = new ModelBuilder(model);
            var input = new Model.Input();
            Tensor kernel0 = new Tensor(8,8,3,16);
            Tensor bias0 = new Tensor(1,1,1,16);
            Tensor kernel1 = new Tensor(4,4,16,32);
            Tensor bias1 = new Tensor(1,1,1,32);

            input.name = "input";
            input.shape = (new TensorShape(1, 20, 21, 3)).ToArray();
            model.inputs.Add(input);

            List<string> outputs = new List<string> { "conv1" };
            builder.model.outputs = outputs;

            builder.Conv2D("conv0", "input", new int[] { 4, 4 }, new int[] { 0, 0, 0, 0 }, kernel0, bias0);
            builder.Conv2D("conv1", "conv0", new int[] { 2, 2 }, new int[] { 0, 0, 0, 0 }, kernel1, bias1);

            //Debug.Log(model);

            m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, model);
        }

        [Test]
        public void TestRepro()
        {
            float[] randomData = new float[1260];

            for ( int i = 0; i < 1260; i++ ) {
                randomData[i] = Random.Range(-100.0f, 100.0f);
            }

            Tensor inputValues = new Tensor(1, 20, 21, 3, randomData);
            m_Worker.Execute(inputValues);
            //Tensor Output = m_Worker.PeekOutput("conv1");
            //Debug.Log(Output);
            inputValues.Dispose();
        }

    }
}
