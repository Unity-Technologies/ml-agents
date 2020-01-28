using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using MLAgents.Sensor;

namespace MLAgents.Tests
{
    public class RayPerceptionSensorTests
    {
        [Test]
        public void TestGetRayAngles()
        {
            var angles = RayPerceptionSensorComponentBase.GetRayAngles(3, 90f);
            var expectedAngles = new [] { 90f, 60f, 120f, 30f, 150f, 0f, 180f };
            Assert.AreEqual(expectedAngles.Length, angles.Length);
            for (var i = 0; i < angles.Length; i++)
            {
                Assert.AreEqual(expectedAngles[i], angles[i], .01);
            }
        }
    }

    public class RayPerception3DTests
    {
        void SetupScene()
        {
            /*
             *      C
             *    S   S
             *      @
             * ^
             * |    S
             * z
             *   x->
             */
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.position = new Vector3(0, 0, 10);
            cube.tag = "wall";

            var sphere1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere1.transform.position = new Vector3(-5, 0, 5);
            sphere1.tag = "ball";

            var sphere2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere2.transform.position = new Vector3(5, 0, 5);
            // No tag for sphere2

            var sphere3 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere3.transform.position = new Vector3(0, 0, -10);
            //Debug.Log($"sphere3.tag={sphere3.tag}");
            //Debug.Log($"cube.id={cube.GetInstanceID()} s1.id={} s2");
        }

        [Test]
        public void TestRaycasts()
        {
            SetupScene();
            var obj = new GameObject("agent");
            var perception = obj.AddComponent<RayPerceptionSensorComponent3D>();

            perception.raysPerDirection = 1;
            perception.maxRayDegrees = 45;
            perception.rayLength = 20;
            perception.detectableTags = new List<string>();
            perception.detectableTags.Add("wall");
            perception.detectableTags.Add("ball");

            var sensor = perception.CreateSensor();

            var expectedObs = (2 * perception.raysPerDirection + 1) * (perception.detectableTags.Count + 2);
            Assert.AreEqual(sensor.GetObservationShape()[0], expectedObs);
            var outputBuffer = new float[expectedObs];

            WriteAdapter writer = new WriteAdapter();
            writer.SetTarget(outputBuffer, sensor.GetObservationShape(), 0);

            var numWritten = sensor.Write(writer);
            Assert.AreEqual(numWritten, expectedObs);

            // Expected hits:
            // ray 0 should hit the "wall" tag at roughly halfway
            // ray 1 should hit the "ball" tag
            // ray 2 should hit but no tag
            // The hit fraction should be the same for rays 1 and
            //
            //Debug.Log($"{outputBuffer[0]} {outputBuffer[1]} {outputBuffer[2]} {outputBuffer[3]}");
            Assert.AreEqual(1.0f, outputBuffer[0]); // hit wall tag
            Assert.AreEqual(0.0f, outputBuffer[1]); // missed ball tag
            Assert.AreEqual(0.0f, outputBuffer[2]); // missed unknown tag
            // Hit is at z=9.0 in world space, ray length is 20
            Assert.That( outputBuffer[3], Is.EqualTo( 9.0f / perception.rayLength ).Within( .0005f ));

            // Spheres are at 5,0,5 and 5,0,-5, so 5*sqrt(2) units from origin
            // Minus 1.0 for the sphere radius to get the length of the hit.
            //Debug.Log($"{outputBuffer[4]} {outputBuffer[5]} {outputBuffer[6]} {outputBuffer[7]}");
            var expectedHitLengthWorldSpace = 5.0f * Mathf.Sqrt(2.0f) - 1.0f;
            Assert.AreEqual(0.0f, outputBuffer[4]); // missed wall tag
            Assert.AreEqual(0.0f, outputBuffer[5]); // missed ball tag
            Assert.AreEqual(0.0f, outputBuffer[6]); // hit unknown tag -> all 0
            Assert.That( outputBuffer[7], Is.EqualTo( expectedHitLengthWorldSpace / perception.rayLength ).Within( .0005f ));

            //Debug.Log($"{outputBuffer[8]} {outputBuffer[9]} {outputBuffer[10]} {outputBuffer[11]}");
            Assert.AreEqual(0.0f, outputBuffer[8]); // missed wall tag
            Assert.AreEqual(1.0f, outputBuffer[9]); // hit ball tag
            Assert.AreEqual(0.0f, outputBuffer[10]); // missed unknown tag
            Assert.That( outputBuffer[11], Is.EqualTo( expectedHitLengthWorldSpace / perception.rayLength ).Within( .0005f ));






        }

    }
}
