using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;
using UnityEditor;

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

            rb1.position = new Vector3(1, 0, 0);
            rb1.rotation = Quaternion.Euler(0, 13.37f, 0);
            rb1.velocity = new Vector3(2, 0, 0);

            Assert.AreEqual(rb1.position, poseExtractor.GetPoseAt(0).position);
            Assert.IsTrue(rb1.rotation == poseExtractor.GetPoseAt(0).rotation);
            Assert.AreEqual(rb1.velocity, poseExtractor.GetLinearVelocityAt(0));
        }

        [Test]
        public void TestTwoBodiesVirtualRoot()
        {
            // * virtualRoot
            // * rootObj
            //   - rb1
            //   * go2
            //     - rb2
            //     - joint
            var virtualRoot = new GameObject("I am vroot");

            var rootObj = new GameObject();
            var rb1 = rootObj.AddComponent<Rigidbody>();

            var go2 = new GameObject();
            var rb2 = go2.AddComponent<Rigidbody>();
            go2.transform.SetParent(rootObj.transform);

            var joint = go2.AddComponent<ConfigurableJoint>();
            joint.connectedBody = rb1;

            var poseExtractor = new RigidBodyPoseExtractor(rb1, null, virtualRoot);
            Assert.AreEqual(3, poseExtractor.NumPoses);

            // "body" 0 has no parent
            Assert.AreEqual(-1, poseExtractor.GetParentIndex(0));

            // body 1 has parent 0
            Assert.AreEqual(0, poseExtractor.GetParentIndex(1));

            var virtualRootPos = new Vector3(0,2,0);
            var virtualRootRot = Quaternion.Euler(0, 42, 0);
            virtualRoot.transform.position = virtualRootPos;
            virtualRoot.transform.rotation = virtualRootRot;

            Assert.AreEqual(virtualRootPos, poseExtractor.GetPoseAt(0).position);
            Assert.IsTrue(virtualRootRot == poseExtractor.GetPoseAt(0).rotation);
            Assert.AreEqual(Vector3.zero, poseExtractor.GetLinearVelocityAt(0));

            // Same as above test, but using index 1
            rb1.position = new Vector3(1, 0, 0);
            rb1.rotation = Quaternion.Euler(0, 13.37f, 0);
            rb1.velocity = new Vector3(2, 0, 0);

            Assert.AreEqual(rb1.position, poseExtractor.GetPoseAt(1).position);
            Assert.IsTrue(rb1.rotation == poseExtractor.GetPoseAt(1).rotation);
            Assert.AreEqual(rb1.velocity, poseExtractor.GetLinearVelocityAt(1));
        }
    }
}
