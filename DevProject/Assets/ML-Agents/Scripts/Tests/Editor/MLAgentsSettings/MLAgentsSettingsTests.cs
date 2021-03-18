using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Editor;


namespace MLAgentsExamples.Tests.Settings
{
    [TestFixture]
    public class MLAgentsSettingsTests
    {
        string EditorBuildSettingsConfigKey = MLAgentsSettingsManager.EditorBuildSettingsConfigKey;
        string tempSettingsRootPath = "Assets/ML-Agents/Scripts/Tests/Editor/MLAgentsSettings";
        MLAgentsSettings storedConfigObject;
        [SetUp]
        public void SetUp()
        {
            if (EditorBuildSettings.TryGetConfigObject(EditorBuildSettingsConfigKey,
                out MLAgentsSettings settingsAsset))
            {
                if (settingsAsset != null)
                {
                    storedConfigObject = settingsAsset;
                    EditorBuildSettings.RemoveConfigObject(EditorBuildSettingsConfigKey);
                }
            }
            MLAgentsSettingsManager.Destroy();
            ClearSettingsAssets();
        }

        [TearDown]
        public void TearDown()
        {
            if (storedConfigObject != null)
            {
                EditorBuildSettings.AddConfigObject(EditorBuildSettingsConfigKey, storedConfigObject, true);
                storedConfigObject = null;
            }
            MLAgentsSettingsManager.Destroy();
            ClearSettingsAssets();
        }

        internal void ClearSettingsAssets()
        {
            var assetsGuids = AssetDatabase.FindAssets("t:MLAgentsSettings", new string[] { tempSettingsRootPath });
            foreach (var guid in assetsGuids)
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                AssetDatabase.DeleteAsset(path);
            }
        }

        [Test]
        public void TestMLAgentsSettingsManager()
        {
            Assert.AreNotEqual(null, MLAgentsSettingsManager.Settings);
            Assert.AreEqual(5004, MLAgentsSettingsManager.Settings.EditorPort); // default port
            MLAgentsSettingsManager.Settings.EditorPort = 6000;
            Assert.AreEqual(6000, MLAgentsSettingsManager.Settings.EditorPort);

            var settingsObject = ScriptableObject.CreateInstance<MLAgentsSettings>();
            settingsObject.EditorPort = 7000;
            var tempSettingsAssetPath = tempSettingsRootPath + "/test.mlagents.settings.asset";
            AssetDatabase.CreateAsset(settingsObject, tempSettingsAssetPath);
            EditorBuildSettings.AddConfigObject(EditorBuildSettingsConfigKey, settingsObject, true);
            // destroy manager instantiated as a side effect by accessing MLAgentsSettings directly without manager
            MLAgentsSettingsManager.Destroy();
            Assert.AreEqual(7000, MLAgentsSettingsManager.Settings.EditorPort);
        }

        // A mock class that can invoke private methods/fields in MLAgentsSettingsProvider
        internal class MockSettingsProvider
        {
            public MLAgentsSettingsProvider Instance
            {
                get
                {
                    return (MLAgentsSettingsProvider)typeof(MLAgentsSettingsProvider).GetField("s_Instance",
                        BindingFlags.Static | BindingFlags.NonPublic).GetValue(null);
                }
            }

            public MLAgentsSettings Settings
            {
                get
                {
                    return (MLAgentsSettings)typeof(MLAgentsSettingsProvider).GetField("m_Settings",
                        BindingFlags.Instance | BindingFlags.NonPublic).GetValue(Instance);
                }
            }

            public void CreateMLAgentsSettingsProvider()
            {
                MLAgentsSettingsProvider.CreateMLAgentsSettingsProvider();
            }

            public void Reinitialize()
            {
                var method = typeof(MLAgentsSettingsProvider).GetMethod("Reinitialize",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                method.Invoke(Instance, null);
            }

            public string[] FindSettingsInProject()
            {
                var method = typeof(MLAgentsSettingsProvider).GetMethod("FindSettingsInProject",
                    BindingFlags.Static | BindingFlags.NonPublic);
                return (string[])method.Invoke(null, null);
            }

            public void CreateNewSettingsAsset(string relativePath)
            {
                var method = typeof(MLAgentsSettingsProvider).GetMethod("CreateNewSettingsAsset",
                    BindingFlags.Static | BindingFlags.NonPublic);
                method.Invoke(null, new object[] { relativePath });
            }
        }

        [Test]
        public void TestMLAgentsSettingsProviderCreateAsset()
        {
            var mockProvider = new MockSettingsProvider();
            mockProvider.CreateMLAgentsSettingsProvider();
            Assert.AreNotEqual(null, mockProvider.Instance);

            // mimic MLAgentsSettingsProvider.OnActivate()
            MLAgentsSettingsManager.OnSettingsChange += mockProvider.Reinitialize;

            mockProvider.Instance.InitializeWithCurrentSettings();
            Assert.AreEqual(0, mockProvider.FindSettingsInProject().Length);

            var tempSettingsAssetPath1 = tempSettingsRootPath + "/test.mlagents.settings.asset";
            mockProvider.CreateNewSettingsAsset(tempSettingsAssetPath1);
            Assert.AreEqual(1, mockProvider.FindSettingsInProject().Length);
            Assert.AreEqual(5004, mockProvider.Settings.EditorPort);
            MLAgentsSettingsManager.Settings.EditorPort = 6000; // change to something not default
            // callback should update the field in provider
            Assert.AreEqual(6000, mockProvider.Settings.EditorPort);

            var tempSettingsAssetPath2 = tempSettingsRootPath + "/test2.mlagents.settings.asset";
            mockProvider.CreateNewSettingsAsset(tempSettingsAssetPath2);
            Assert.AreEqual(2, mockProvider.FindSettingsInProject().Length);
            // manager should set to the new (default) one, not the previous modified one
            Assert.AreEqual(5004, MLAgentsSettingsManager.Settings.EditorPort);

            // mimic MLAgentsSettingsProvider.OnDeactivate()
            MLAgentsSettingsManager.OnSettingsChange -= mockProvider.Reinitialize;
            mockProvider.Instance.Dispose();
        }

        [Test]
        public void TestMLAgentsSettingsProviderLoadAsset()
        {
            var mockProvider = new MockSettingsProvider();
            var tempSettingsAssetPath1 = tempSettingsRootPath + "/test.mlagents.settings.asset";
            mockProvider.CreateNewSettingsAsset(tempSettingsAssetPath1);
            MLAgentsSettingsManager.Settings.EditorPort = 8000; // change to something not default

            mockProvider.Instance?.Dispose();
            MLAgentsSettingsManager.Destroy();

            mockProvider.CreateMLAgentsSettingsProvider();
            Assert.AreEqual(8000, MLAgentsSettingsManager.Settings.EditorPort);
        }
    }
}
