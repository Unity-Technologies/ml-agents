using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Utils.Tests;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class RenderTextureSensorComponentTest
    {
        [Test]
        public void TestRenderTextureSensorComponent()
        {
            foreach (var grayscale in new[] { true, false })
            {
                foreach (SensorCompressionType compression in Enum.GetValues(typeof(SensorCompressionType)))
                {
                    var width = 24;
                    var height = 16;
                    var texture = new RenderTexture(width, height, 0);

                    var agentGameObj = new GameObject("agent");

                    var renderTexComponent = agentGameObj.AddComponent<RenderTextureSensorComponent>();
                    renderTexComponent.RenderTexture = texture;
                    renderTexComponent.Grayscale = grayscale;
                    renderTexComponent.CompressionType = compression;

                    var expectedShape = new InplaceArray<int>(grayscale ? 1 : 3, height, width);

                    var sensor = renderTexComponent.CreateSensors()[0];
                    Assert.AreEqual(expectedShape, sensor.GetObservationSpec().Shape);
                    Assert.AreEqual(typeof(RenderTextureSensor), sensor.GetType());
                }
            }
        }

        [Test]
        public void DisableEnableAgent([Values(1, 3)] int observationStacks)
        {
            var flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var texture = new RenderTexture(24, 16, 0);
            var parentGameObject = new GameObject("OuterAgent");
            var childGameObject = new GameObject("ChildAgent");
            try
            {
                var renderTexComponent = childGameObject.AddComponent<RenderTextureSensorComponent>();
                renderTexComponent.RenderTexture = texture;
                renderTexComponent.ObservationStacks = observationStacks;

                var (_, childAgent) = TestAgent.CreateNestedAgents(parentGameObject, childGameObject);

                var componentSensors = (List<RenderTextureSensor>)typeof(RenderTextureSensorComponent).GetField("m_Sensors", flags).GetValue(renderTexComponent);
                Assert.AreEqual(2, componentSensors.Count);
                var parentSensor = componentSensors[0];

                for (var i = 0; i < 5; i++)
                {
                    // Agent.OnDisable() disposes its sensors; Agent.OnEnable() re-initializes and calls CreateSensors() again.
                    childAgent.enabled = false;
                    childAgent.enabled = true;
                    renderTexComponent.CompressionType = SensorCompressionType.None;
                    Assert.AreEqual(2, componentSensors.Count, "disposed sensors should be pruned, not retained");
                }

                Assert.IsFalse(componentSensors.Exists(s => s.IsDisposed));
                Assert.Contains(parentSensor, componentSensors, "the parent Agent's sensor should not be disposed");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(parentGameObject);
                UnityEngine.Object.DestroyImmediate(texture);
                if (Academy.IsInitialized)
                {
                    Academy.Instance.Dispose();
                }
            }
        }
    }
}
