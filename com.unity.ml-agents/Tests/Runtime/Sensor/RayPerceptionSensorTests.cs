using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Unity.MLAgents.Sensors;
using UnityEngine.TestTools;

namespace Unity.MLAgents.Tests
{
    public class RayPerceptionSensorTests
    {
        [Test]
        public void TestGetRayAngles()
        {
            var angles = RayPerceptionSensorComponentBase.GetRayAngles(3, 90f);
            var expectedAngles = new[] { 90f, 60f, 120f, 30f, 150f, 0f, 180f };
            Assert.AreEqual(expectedAngles.Length, angles.Length);
            for (var i = 0; i < angles.Length; i++)
            {
                Assert.AreEqual(expectedAngles[i], angles[i], .01);
            }
        }
    }

    public class RayPerception3DTests
    {
        [Test]
        public void TestDefaultLayersAreNegativeFive()
        {
#if MLA_UNITY_PHYSICS_MODULE
            Assert.IsTrue(Physics.DefaultRaycastLayers == -5);
#endif
#if MLA_UNITY_PHYSICS2D_MODULE
            Assert.IsTrue(Physics2D.DefaultRaycastLayers == -5);
#endif
        }

#if MLA_UNITY_PHYSICS_MODULE
        // Use built-in tags
        const string k_CubeTag = "Player";
        const string k_SphereTag = "Respawn";

        [TearDown]
        public void RemoveGameObjects()
        {
            var objects = GameObject.FindObjectsOfType<GameObject>();
            foreach (var o in objects)
            {
                UnityEngine.Object.DestroyImmediate(o);
            }
        }

        void SetupScene()
        {
            /* Creates game objects in the world for testing.
             *   C is a cube
             *   S are spheres
             *   @ is the agent (at the origin)
             * Each space or line is 5 world units, +x is right, +z is up
             *
             *      C
             *    S   S
             *      @
             *
             *      S
             */
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = new Vector3(0, 0, 10);
            cube.tag = k_CubeTag;
            cube.name = "cube";

            var sphere1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere1.transform.position = new Vector3(-5, 0, 5);
            sphere1.tag = k_SphereTag;
            sphere1.name = "sphere1";

            var sphere2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere2.transform.position = new Vector3(5, 0, 5);
            // No tag for sphere2
            sphere2.name = "sphere2";

            var sphere3 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere3.transform.position = new Vector3(0, 0, -10);
            sphere3.tag = k_SphereTag;
            sphere3.name = "sphere3";


            Physics.SyncTransforms();
        }

        [Test]
        public void TestRaycasts()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.RaysPerDirection = 1;
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add(k_CubeTag);
            perception.DetectableTags.Add(k_SphereTag);

            var radii = new[] { 0f, .5f };
            foreach (var castRadius in radii)
            {
                perception.SphereCastRadius = castRadius;
                var sensor = perception.CreateSensors()[0];
                sensor.Update();

                var expectedObs = (2 * perception.RaysPerDirection + 1) * (perception.DetectableTags.Count + 2);
                Assert.AreEqual(sensor.GetObservationSpec().Shape[0], expectedObs);
                var outputBuffer = new float[expectedObs];

                ObservationWriter writer = new ObservationWriter();
                writer.SetTarget(outputBuffer, sensor.GetObservationSpec(), 0);

                var numWritten = sensor.Write(writer);
                Assert.AreEqual(numWritten, expectedObs);

                // Expected hits:
                // ray 0 should hit the cube at roughly halfway
                // ray 1 should hit a sphere but no tag
                // ray 2 should hit a sphere with the k_SphereTag tag
                // The hit fraction should be the same for rays 1 and
                //
                Assert.AreEqual(1.0f, outputBuffer[0]); // hit cube
                Assert.AreEqual(0.0f, outputBuffer[1]); // missed sphere
                Assert.AreEqual(0.0f, outputBuffer[2]); // missed unknown tag

                // Hit is at z=9.0 in world space, ray length is 20
                Assert.That(
                    outputBuffer[3], Is.EqualTo((9.5f - castRadius) / perception.RayLength).Within(.0005f)
                );

                // Spheres are at 5,0,5 and 5,0,-5, so 5*sqrt(2) units from origin
                // Minus 1.0 for the sphere radius to get the length of the hit.
                var expectedHitLengthWorldSpace = 5.0f * Mathf.Sqrt(2.0f) - 0.5f - castRadius;
                Assert.AreEqual(0.0f, outputBuffer[4]); // missed cube
                Assert.AreEqual(0.0f, outputBuffer[5]); // missed sphere
                Assert.AreEqual(0.0f, outputBuffer[6]); // hit unknown tag -> all 0
                Assert.That(
                    outputBuffer[7], Is.EqualTo(expectedHitLengthWorldSpace / perception.RayLength).Within(.0005f)
                );

                Assert.AreEqual(0.0f, outputBuffer[8]); // missed cube
                Assert.AreEqual(1.0f, outputBuffer[9]); // hit sphere
                Assert.AreEqual(0.0f, outputBuffer[10]); // missed unknown tag
                Assert.That(
                    outputBuffer[11], Is.EqualTo(expectedHitLengthWorldSpace / perception.RayLength).Within(.0005f)
                );
            }
        }

        [Test]
        public void TestRaycastMiss()
        {
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.RaysPerDirection = 0;
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add(k_CubeTag);
            perception.DetectableTags.Add(k_SphereTag);

            var sensor = perception.CreateSensors()[0];
            sensor.Update();
            var expectedObs = (2 * perception.RaysPerDirection + 1) * (perception.DetectableTags.Count + 2);
            Assert.AreEqual(sensor.GetObservationSpec().Shape[0], expectedObs);
            var outputBuffer = new float[expectedObs];

            ObservationWriter writer = new ObservationWriter();
            writer.SetTarget(outputBuffer, sensor.GetObservationSpec(), 0);

            var numWritten = sensor.Write(writer);
            Assert.AreEqual(numWritten, expectedObs);

            // Everything missed
            Assert.AreEqual(new float[] { 0, 0, 1, 1 }, outputBuffer);
        }

        [Test]
        public void TestRayFilter()
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = new Vector3(0, 0, 10);
            cube.tag = k_CubeTag;
            cube.name = "cubeFar";

            var cubeFiltered = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cubeFiltered.transform.position = new Vector3(0, 0, 5);
            cubeFiltered.tag = k_CubeTag;
            cubeFiltered.name = "cubeNear";
            cubeFiltered.layer = 7;

            Physics.SyncTransforms();

            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();
            perception.RaysPerDirection = 0;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();

            var filterCubeLayers = new[] { false, true };
            foreach (var filterCubeLayer in filterCubeLayers)
            {
                // Set the layer mask to either the default, or one that ignores the close cube's layer
                var layerMask = Physics.DefaultRaycastLayers;
                if (filterCubeLayer)
                {
                    layerMask &= ~(1 << cubeFiltered.layer);
                }
                perception.RayLayerMask = layerMask;

                var sensor = perception.CreateSensors()[0];
                sensor.Update();
                var expectedObs = (2 * perception.RaysPerDirection + 1) * (perception.DetectableTags.Count + 2);
                Assert.AreEqual(sensor.GetObservationSpec().Shape[0], expectedObs);
                var outputBuffer = new float[expectedObs];

                ObservationWriter writer = new ObservationWriter();
                writer.SetTarget(outputBuffer, sensor.GetObservationSpec(), 0);

                var numWritten = sensor.Write(writer);
                Assert.AreEqual(numWritten, expectedObs);

                if (filterCubeLayer)
                {
                    // Hit the far cube because close was filtered.
                    Assert.That(outputBuffer[outputBuffer.Length - 1],
                        Is.EqualTo((9.5f - perception.SphereCastRadius) / perception.RayLength).Within(.0005f)
                    );
                }
                else
                {
                    // Hit the close cube because not filtered.
                    Assert.That(outputBuffer[outputBuffer.Length - 1],
                        Is.EqualTo((4.5f - perception.SphereCastRadius) / perception.RayLength).Within(.0005f)
                    );
                }
            }
        }

        [Test]
        public void TestRaycastsScaled()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();
            obj.transform.localScale = new Vector3(2, 2, 2);

            perception.RaysPerDirection = 0;
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add(k_CubeTag);

            var radii = new[] { 0f, .5f };
            foreach (var castRadius in radii)
            {
                perception.SphereCastRadius = castRadius;
                var sensor = perception.CreateSensors()[0];
                sensor.Update();

                var expectedObs = (2 * perception.RaysPerDirection + 1) * (perception.DetectableTags.Count + 2);
                Assert.AreEqual(sensor.GetObservationSpec().Shape[0], expectedObs);
                var outputBuffer = new float[expectedObs];

                ObservationWriter writer = new ObservationWriter();
                writer.SetTarget(outputBuffer, sensor.GetObservationSpec(), 0);

                var numWritten = sensor.Write(writer);
                Assert.AreEqual(numWritten, expectedObs);

                // Expected hits:
                // ray 0 should hit the cube at roughly 1/4 way
                //
                Assert.AreEqual(1.0f, outputBuffer[0]); // hit cube
                Assert.AreEqual(0.0f, outputBuffer[1]); // missed unknown tag

                // Hit is at z=9.0 in world space, ray length was 20
                // But scale increases the cast size and the ray length
                var scaledRayLength = 2 * perception.RayLength;
                var scaledCastRadius = 2 * castRadius;
                Assert.That(
                    outputBuffer[2], Is.EqualTo((9.5f - scaledCastRadius) / scaledRayLength).Within(.0005f)
                );
            }
        }

        [Test]
        public void TestRayZeroLength()
        {
            // Place the cube touching the origin
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = new Vector3(0, 0, .5f);
            cube.tag = k_CubeTag;

            Physics.SyncTransforms();

            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();
            perception.RaysPerDirection = 0;
            perception.RayLength = 0.0f;
            perception.SphereCastRadius = .5f;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add(k_CubeTag);

            {
                // Set the layer mask to either the default, or one that ignores the close cube's layer

                var sensor = perception.CreateSensors()[0];
                sensor.Update();
                var expectedObs = (2 * perception.RaysPerDirection + 1) * (perception.DetectableTags.Count + 2);
                Assert.AreEqual(sensor.GetObservationSpec().Shape[0], expectedObs);
                var outputBuffer = new float[expectedObs];

                ObservationWriter writer = new ObservationWriter();
                writer.SetTarget(outputBuffer, sensor.GetObservationSpec(), 0);

                var numWritten = sensor.Write(writer);
                Assert.AreEqual(numWritten, expectedObs);

                // hit fraction is arbitrary but should be finite in [0,1]
                Assert.GreaterOrEqual(outputBuffer[2], 0.0f);
                Assert.LessOrEqual(outputBuffer[2], 1.0f);
            }
        }

        [Test]
        public void TestStaticPerceive()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.RaysPerDirection = 0; // single ray
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add(k_CubeTag);
            perception.DetectableTags.Add(k_SphereTag);

            var radii = new[] { 0f, .5f };
            foreach (var castRadius in radii)
            {
                perception.SphereCastRadius = castRadius;
                var castInput = perception.GetRayPerceptionInput();
                var castOutput = RayPerceptionSensor.Perceive(castInput);

                Assert.AreEqual(1, castOutput.RayOutputs.Length);

                // Expected to hit the cube
                Assert.AreEqual("cube", castOutput.RayOutputs[0].HitGameObject.name);
                Assert.AreEqual(0, castOutput.RayOutputs[0].HitTagIndex);
            }
        }

        [Test]
        public void TestStaticPerceiveInvalidTags()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.RaysPerDirection = 0; // single ray
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = new List<string>();
            perception.DetectableTags.Add("Bad tag");
            perception.DetectableTags.Add(null);
            perception.DetectableTags.Add("");
            perception.DetectableTags.Add(k_CubeTag);

            var radii = new[] { 0f, .5f };
            foreach (var castRadius in radii)
            {
                perception.SphereCastRadius = castRadius;
                var castInput = perception.GetRayPerceptionInput();

                // There's no clean way that I can find to check for a defined tag without
                // logging an error.
                LogAssert.Expect(LogType.Error, "Tag: Bad tag is not defined.");
                var castOutput = RayPerceptionSensor.Perceive(castInput);

                Assert.AreEqual(1, castOutput.RayOutputs.Length);

                // Expected to hit the cube
                Assert.AreEqual("cube", castOutput.RayOutputs[0].HitGameObject.name);
                Assert.AreEqual(3, castOutput.RayOutputs[0].HitTagIndex);
            }
        }

        [Test]
        public void TestStaticPerceiveNoTags()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.RaysPerDirection = 0; // single ray
            perception.MaxRayDegrees = 45;
            perception.RayLength = 20;
            perception.DetectableTags = null;

            var radii = new[] { 0f, .5f };
            foreach (var castRadius in radii)
            {
                perception.SphereCastRadius = castRadius;
                var castInput = perception.GetRayPerceptionInput();
                var castOutput = RayPerceptionSensor.Perceive(castInput);

                Assert.AreEqual(1, castOutput.RayOutputs.Length);

                // Expected to hit the cube
                Assert.AreEqual("cube", castOutput.RayOutputs[0].HitGameObject.name);
                Assert.AreEqual(-1, castOutput.RayOutputs[0].HitTagIndex);
            }
        }

        [Test]
        public void TestCreateDefault()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            Assert.DoesNotThrow(() =>
            {
                perception.CreateSensors();
            });
        }
#endif
    }
}
