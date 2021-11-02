using System;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Assertions.Comparers;

namespace Unity.MLAgents.Tests
{
    public class FloatThresholdComparer : IEqualityComparer<float>
    {
        private readonly float _threshold;
        public FloatThresholdComparer(float threshold)
        {
            _threshold = threshold;
        }

        public bool Equals(float x, float y)
        {
            return Math.Abs(x - y) < _threshold;
        }

        public int GetHashCode(float f)
        {
            throw new NotImplementedException("Unable to generate a hash code for threshold floats, do not use this method");
        }
    }

    [TestFixture]
    public class ModelRunnerTest
    {
        const string k_hybrid_ONNX_recurr_v2 = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/hybrid0vis8vec_2c_2_3d_v2_0.onnx";

        const string k_continuousONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action_v1_0.onnx";
        const string k_discreteONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/discrete1vis0vec_2_3action_obsolete_recurr_v1_0.onnx";
        const string k_hybridONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/hybrid0vis53vec_3c_2daction_v1_0.onnx";
        const string k_continuousNNPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action_deprecated_v1_0.nn";
        const string k_discreteNNPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/discrete1vis0vec_2_3action_recurr_deprecated_v1_0.nn";
        // models with deterministic action tensors
        private const string k_deter_discreteNNPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/deterDiscrete1obs3action_v2_0.onnx";
        private const string k_deter_continuousNNPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/deterContinuous2vis8vec2action_v2_0.onnx";

        NNModel hybridONNXModelV2;
        NNModel continuousONNXModel;
        NNModel discreteONNXModel;
        NNModel hybridONNXModel;
        NNModel continuousNNModel;
        NNModel discreteNNModel;
        NNModel deterDiscreteNNModel;
        NNModel deterContinuousNNModel;
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

        ActionSpec GetHybrid0vis53vec_3c_2dActionSpec()
        {
            return new ActionSpec(3, new[] { 2 });
        }

        [SetUp]
        public void SetUp()
        {
            hybridONNXModelV2 = (NNModel)AssetDatabase.LoadAssetAtPath(k_hybrid_ONNX_recurr_v2, typeof(NNModel));

            continuousONNXModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_continuousONNXPath, typeof(NNModel));
            discreteONNXModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_discreteONNXPath, typeof(NNModel));
            hybridONNXModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_hybridONNXPath, typeof(NNModel));
            continuousNNModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_continuousNNPath, typeof(NNModel));
            discreteNNModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_discreteNNPath, typeof(NNModel));
            deterDiscreteNNModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_deter_discreteNNPath, typeof(NNModel));
            deterContinuousNNModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_deter_continuousNNPath, typeof(NNModel));
            var go = new GameObject("SensorA");
            sensor_21_20_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_21_20_3.Sensor = new Test3DSensor("SensorA", 21, 20, 3);
            sensor_20_22_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_20_22_3.Sensor = new Test3DSensor("SensorB", 20, 22, 3);
        }

        [Test]
        public void TestModelExist()
        {
            Assert.IsNotNull(continuousONNXModel);
            Assert.IsNotNull(discreteONNXModel);
            Assert.IsNotNull(hybridONNXModel);
            Assert.IsNotNull(continuousNNModel);
            Assert.IsNotNull(discreteNNModel);
            Assert.IsNotNull(hybridONNXModelV2);
            Assert.IsNotNull(deterDiscreteNNModel);
            Assert.IsNotNull(deterContinuousNNModel);
        }

        [Test]
        public void TestCreation()
        {
            var inferenceDevice = InferenceDevice.Burst;
            var modelRunner = new ModelRunner(continuousONNXModel, GetContinuous2vis8vec2actionActionSpec(), inferenceDevice);
            modelRunner.Dispose();
            Assert.Throws<UnityAgentsException>(() =>
            {
                // Cannot load a model trained with 1.x that has an LSTM
                modelRunner = new ModelRunner(discreteONNXModel, GetDiscrete1vis0vec_2_3action_recurrModelActionSpec(), inferenceDevice);
                modelRunner.Dispose();
            });
            modelRunner = new ModelRunner(hybridONNXModel, GetHybrid0vis53vec_3c_2dActionSpec(), inferenceDevice);
            modelRunner.Dispose();
            modelRunner = new ModelRunner(continuousNNModel, GetContinuous2vis8vec2actionActionSpec(), inferenceDevice);
            modelRunner.Dispose();

            Assert.Throws<UnityAgentsException>(() =>
            {
                // Cannot load a model trained with 1.x that has an LSTM
                modelRunner = new ModelRunner(discreteNNModel, GetDiscrete1vis0vec_2_3action_recurrModelActionSpec(), inferenceDevice);
                modelRunner.Dispose();
            });
            // This one was trained with 2.0 so it should not raise an error:
            modelRunner = new ModelRunner(hybridONNXModelV2, new ActionSpec(2, new[] { 2, 3 }), inferenceDevice);
            modelRunner.Dispose();

            // V2.0 Model that has serialized deterministic action tensors, discrete
            modelRunner = new ModelRunner(deterDiscreteNNModel, new ActionSpec(0, new[] { 7 }), inferenceDevice);
            modelRunner.Dispose();
            // V2.0 Model that has serialized deterministic action tensors, continuous
            modelRunner = new ModelRunner(deterContinuousNNModel,
                GetContinuous2vis8vec2actionActionSpec(), inferenceDevice,
                stochasticInference: false);
            modelRunner.Dispose();
        }

        [Test]
        public void TestHasModel()
        {
            var modelRunner = new ModelRunner(continuousONNXModel, GetContinuous2vis8vec2actionActionSpec(), InferenceDevice.CPU);
            Assert.True(modelRunner.HasModel(continuousONNXModel, InferenceDevice.CPU));
            Assert.False(modelRunner.HasModel(continuousONNXModel, InferenceDevice.GPU));
            Assert.False(modelRunner.HasModel(discreteONNXModel, InferenceDevice.CPU));
            modelRunner.Dispose();
        }

        [Test]
        public void TestRunModel()
        {
            var actionSpec = GetContinuous2vis8vec2actionActionSpec();
            var modelRunner = new ModelRunner(continuousONNXModel, actionSpec, InferenceDevice.Burst);
            var sensor_8 = new Sensors.VectorSensor(8, "VectorSensor8");
            var info1 = new AgentInfo();
            info1.episodeId = 1;
            modelRunner.PutObservations(info1, new[] {
                sensor_8,
                sensor_21_20_3.CreateSensors()[0],
                sensor_20_22_3.CreateSensors()[0] }.ToList());
            var info2 = new AgentInfo();
            info2.episodeId = 2;
            modelRunner.PutObservations(info2, new[] {
                sensor_8,
                sensor_21_20_3.CreateSensors()[0],
                sensor_20_22_3.CreateSensors()[0] }.ToList());

            modelRunner.DecideBatch();

            Assert.IsFalse(modelRunner.GetAction(1).Equals(ActionBuffers.Empty));
            Assert.IsFalse(modelRunner.GetAction(2).Equals(ActionBuffers.Empty));
            Assert.IsTrue(modelRunner.GetAction(3).Equals(ActionBuffers.Empty));
            Assert.AreEqual(actionSpec.NumDiscreteActions, modelRunner.GetAction(1).DiscreteActions.Length);
            modelRunner.Dispose();
        }


        [Test]
        public void TestRunModel_deterministic()
        {
            var actionSpec = GetContinuous2vis8vec2actionActionSpec();
            var modelRunner = new ModelRunner(deterContinuousNNModel, actionSpec, InferenceDevice.Burst);
            var sensor_8 = new Sensors.VectorSensor(8, "VectorSensor8");
            var info1 = new AgentInfo();
            var obs = new[]
            {
                sensor_8,
                sensor_21_20_3.CreateSensors()[0],
                sensor_20_22_3.CreateSensors()[0]
            }.ToList();
            info1.episodeId = 1;
            modelRunner.PutObservations(info1, obs);
            modelRunner.DecideBatch();
            var stochAction1 = (float[])modelRunner.GetAction(1).ContinuousActions.Array.Clone();

            modelRunner.PutObservations(info1, obs);
            modelRunner.DecideBatch();
            var stochAction2 = (float[])modelRunner.GetAction(1).ContinuousActions.Array.Clone();
            // Stochastic action selection should output randomly different action values with same obs
            Assert.IsFalse(Enumerable.SequenceEqual(stochAction1, stochAction2, new FloatThresholdComparer(0.001f)));


            var deterModelRunner = new ModelRunner(deterContinuousNNModel, actionSpec, InferenceDevice.Burst,
                stochasticInference: false);
            info1.episodeId = 1;
            deterModelRunner.PutObservations(info1, obs);
            deterModelRunner.DecideBatch();
            var deterAction1 = (float[])deterModelRunner.GetAction(1).ContinuousActions.Array.Clone();

            deterModelRunner.PutObservations(info1, obs);
            deterModelRunner.DecideBatch();
            var deterAction2 = (float[])deterModelRunner.GetAction(1).ContinuousActions.Array.Clone();
            // Deterministic action selection should output same action everytime
            Assert.IsTrue(Enumerable.SequenceEqual(deterAction1, deterAction2, new FloatThresholdComparer(0.001f)));
            modelRunner.Dispose();
        }
    }
}
