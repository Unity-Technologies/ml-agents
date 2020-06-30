using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class RigidBodyPoseExtractorTests
    {
        [TearDown]
        public void RemoveGameObjects()
        {
            var objects = GameObject.FindObjectsOfType<GameObject>();
            foreach (var o in objects)
            {
                UnityEngine.Object.DestroyImmediate(o);
            }
        }

        [Test]
        public void TestNullRoot()
        {
            var poseExtractor = new RigidBodyPoseExtractor(null);
            // These should be no-ops
            poseExtractor.UpdateLocalSpacePoses();
            poseExtractor.UpdateModelSpacePoses();

            Assert.AreEqual(0, poseExtractor.NumPoses);
        }

        [Test]
        public void TestSingleBody()
        {
            var go = new GameObject();
            var rootRb = go.AddComponent<Rigidbody>();
            var poseExtractor = new RigidBodyPoseExtractor(rootRb);
            Assert.AreEqual(1, poseExtractor.NumPoses);
        }

        [Test]
        public void TestTwoBodies()
        {
            // * rootObj
            //   - rb1
            //   * go2
            //     - rb2
            //     - joint
            var rootObj = new GameObject();
            var rb1 = rootObj.AddComponent<Rigidbody>();

            var go2 = new GameObject();
            var rb2 = go2.AddComponent<Rigidbody>();
            go2.transform.SetParent(rootObj.transform);

            var joint = go2.AddComponent<ConfigurableJoint>();
            joint.connectedBody = rb1;

            var poseExtractor = new RigidBodyPoseExtractor(rb1);
            Assert.AreEqual(2, poseExtractor.NumPoses);
        }
    }
}
