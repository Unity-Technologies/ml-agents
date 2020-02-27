using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Barracuda;
using MLAgents.InferenceBrain;
using System.Linq;

namespace MLAgents.Tests
{

[TestFixture]
    public class ParameterLoaderTest : MonoBehaviour
    {

        const string k_ballanceballModelPath = "Packages/com.unity.ml-agents/Tests/Editor/Ressources/test_3dball.nn";
        const string k_foodcollectorModelPath = "Packages/com.unity.ml-agents/Tests/Editor/Ressources/test_foodcollector.nn";
        NNModel ballanceballModel;
        NNModel foodcollectorModel;

        [SetUp]
        public void SetUp()
        {
            ballanceballModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_ballanceballModelPath, typeof(NNModel));
            foodcollectorModel = (NNModel)AssetDatabase.LoadAssetAtPath(k_foodcollectorModelPath, typeof(NNModel));
        }

        [Test]
        public void TestModelExist()
        {
            Assert.IsNotNull(ballanceballModel);
            Assert.IsNotNull(foodcollectorModel);
        }

        [Test]
        public void TestGetInputTensors()
        {
            var model = ModelLoader.Load(ballanceballModel);
            var inputTensors = BarracudaModelParamLoader.GetInputTensors(model);
            var inputNames = inputTensors.Select(x => x.name).ToList();
            Assert.Contains(TensorNames.RandomNormalEpsilonPlaceholder, inputNames);
            Assert.Contains(TensorNames.VectorObservationPlaceholder, inputNames);
            Assert.AreEqual(2, inputNames.Count);

        }


    }
}
