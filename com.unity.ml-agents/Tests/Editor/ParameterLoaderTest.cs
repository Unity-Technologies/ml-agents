using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Barracuda;
using MLAgents.InferenceBrain;
using MLAgents.Sensors;
using System.Linq;

namespace MLAgents.Tests
{

    public class Test3DSensorComponent : SensorComponent{
        public ISensor Sensor;

        public override ISensor CreateSensor(){
            return Sensor;
        }

        public override int[] GetObservationShape(){
            return Sensor.GetObservationShape();
        }
    }
    public class Test3DSensor : ISensor
    {
        int m_Width;
        int m_Height;
        int m_Channels;
        string m_Name;

        public Test3DSensor(string name, int width, int height, int channels){
            m_Width = width;
            m_Height = height;
            m_Channels = channels;
            m_Name = name;
        }

        public int[] GetObservationShape(){
            return new int[] {m_Height, m_Width, m_Channels };
        }

        public int Write(WriteAdapter adapter)
        {
            return m_Width * m_Height * m_Width;
        }

        public byte[] GetCompressedObservation()
        {
            return new byte[0];
        }
        public void Update(){}

        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        public string GetName(){
            return m_Name;
        }
    }

[TestFixture]
    public class ParameterLoaderTest : MonoBehaviour
    {
        const string k_continuous2vis8vec2actionPath = "Packages/com.unity.ml-agents/Tests/Editor/Ressources/continuous2vis8vec2action.nn";
        const string k_foodcollectorModelPath = "Packages/com.unity.ml-agents/Tests/Editor/Ressources/test_foodcollector.nn";
        NNModel continuous2vis8vec2actionModel;
        NNModel foodcollectorModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;

        private BrainParameters GetContinuous2vis8vec2actionModelBrainParameters(){
            var validBrainParameters = new BrainParameters();
            validBrainParameters.vectorObservationSize = 8;
            validBrainParameters.vectorActionSize = new int[] { 2 };
            validBrainParameters.numStackedVectorObservations = 1;
            validBrainParameters.vectorActionSpaceType = SpaceType.Continuous;
            return validBrainParameters;
        }

        [SetUp]
        public void SetUp()
        {
            continuous2vis8vec2actionModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_continuous2vis8vec2actionPath, typeof(NNModel));
            foodcollectorModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_foodcollectorModelPath, typeof(NNModel));
            var go = new GameObject("SensorA");
            sensor_21_20_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_21_20_3.Sensor = new Test3DSensor("SensorA", 21, 20, 3);
            sensor_20_22_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_20_22_3.Sensor = new Test3DSensor("SensorA", 20, 22, 3);

        }

        [Test]
        public void TestModelExist()
        {
            Assert.IsNotNull(continuous2vis8vec2actionModel);
            Assert.IsNotNull(foodcollectorModel);
        }

        [Test]
        public void TestGetInputTensors()
        {
            var model = ModelLoader.Load(continuous2vis8vec2actionModel);
            var inputTensors = BarracudaModelParamLoader.GetInputTensors(model);
            var inputNames = inputTensors.Select(x => x.name).ToList();
            // Model should contain 3 inputs : vector, visual 1 and visual 2
            Assert.AreEqual(3, inputNames.Count);
            Assert.Contains(TensorNames.VectorObservationPlaceholder, inputNames);
            Assert.Contains(TensorNames.VisualObservationPlaceholderPrefix + "0", inputNames);
            Assert.Contains(TensorNames.VisualObservationPlaceholderPrefix + "1", inputNames);

            Assert.AreEqual(2, BarracudaModelParamLoader.GetNumVisualInputs(model));
        }

        [Test]
        public void TestGetOutputTensors()
        {
            var model = ModelLoader.Load(continuous2vis8vec2actionModel);
            var outputNames = BarracudaModelParamLoader.GetOutputNames(model);
            Assert.Contains(TensorNames.ActionOutput, outputNames);
        }

        [Test]
        public void TestCheckModelValid(){
            var model = ModelLoader.Load(continuous2vis8vec2actionModel);
            var validBrainParameters = GetContinuous2vis8vec2actionModelBrainParameters();

            var errors = BarracudaModelParamLoader.CheckModel(model, validBrainParameters, new SensorComponent[] { sensor_21_20_3, sensor_20_22_3 });
            Assert.AreEqual(0, errors.Count()); // There should not be any errors
        }

        [Test]
        public void TestCheckModelThrowsVectorObservation(){
            var model = ModelLoader.Load(continuous2vis8vec2actionModel);

            var brainParameters = GetContinuous2vis8vec2actionModelBrainParameters();
            brainParameters.vectorObservationSize = 9; // Invalid observation
            var errors = BarracudaModelParamLoader.CheckModel(model, brainParameters, new SensorComponent[] { sensor_21_20_3, sensor_20_22_3 });
            Assert.AreEqual(1, errors.Count()); // There should not only one error

            brainParameters = GetContinuous2vis8vec2actionModelBrainParameters();
            brainParameters.numStackedVectorObservations = 2;// Invalid stacking
            errors = BarracudaModelParamLoader.CheckModel(model, brainParameters, new SensorComponent[] { sensor_21_20_3, sensor_20_22_3 });
            Assert.AreEqual(1, errors.Count()); // There should not only one error
        }

        [Test]
        public void TestCheckModelThrowsAction(){
            var model = ModelLoader.Load(continuous2vis8vec2actionModel);

            var brainParameters = GetContinuous2vis8vec2actionModelBrainParameters();
            brainParameters.vectorActionSize = new int[] { 3 }; // Invalid observation
            var errors = BarracudaModelParamLoader.CheckModel(model, brainParameters, new SensorComponent[] { sensor_21_20_3, sensor_20_22_3 });
            Assert.AreEqual(1, errors.Count()); // There should not only one error

            brainParameters = GetContinuous2vis8vec2actionModelBrainParameters();
            brainParameters.vectorActionSpaceType = SpaceType.Discrete;// Invalid stacking
            errors = BarracudaModelParamLoader.CheckModel(model, brainParameters, new SensorComponent[] { sensor_21_20_3, sensor_20_22_3 });
            Assert.AreEqual(1, errors.Count()); // There should not only one error
        }

    }
}
