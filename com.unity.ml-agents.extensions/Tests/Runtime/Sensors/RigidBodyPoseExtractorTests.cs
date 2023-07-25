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

            // Also pass the GameObject
            poseExtractor = new RigidBodyPoseExtractor(rootRb, go);
            Assert.AreEqual(1, poseExtractor.NumPoses);
        }

        [Test]
        public void TestNoBodiesFound()
        {
            // Check that if we can't find any bodies under the game object, we get an empty extractor
            var gameObj = new GameObject();
            var rootRb = gameObj.AddComponent<Rigidbody>();
            var otherGameObj = new GameObject();
            var poseExtractor = new RigidBodyPoseExtractor(rootRb, otherGameObj);
            Assert.AreEqual(0, poseExtractor.NumPoses);

            // Add an RB under the other GameObject. Constructor will find a rigid body, but not the root.
            otherGameObj.AddComponent<Rigidbody>();
            poseExtractor = new RigidBodyPoseExtractor(rootRb, otherGameObj);
            Assert.AreEqual(0, poseExtractor.NumPoses);
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

            // Check DisplayNodes gives expected results
            var displayNodes = poseExtractor.GetDisplayNodes();
            Assert.AreEqual(2, displayNodes.Count);
            Assert.AreEqual(rb1, displayNodes[0].NodeObject);
            Assert.AreEqual(false, displayNodes[0].Enabled);

            Assert.AreEqual(rb2, displayNodes[1].NodeObject);
            Assert.AreEqual(true, displayNodes[1].Enabled);
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
            go2.AddComponent<Rigidbody>();
            go2.transform.SetParent(rootObj.transform);

            var joint = go2.AddComponent<ConfigurableJoint>();
            joint.connectedBody = rb1;

            var poseExtractor = new RigidBodyPoseExtractor(rb1, null, virtualRoot);
            Assert.AreEqual(3, poseExtractor.NumPoses);

            // "body" 0 has no parent
            Assert.AreEqual(-1, poseExtractor.GetParentIndex(0));

            // body 1 has parent 0
            Assert.AreEqual(0, poseExtractor.GetParentIndex(1));

            var virtualRootPos = new Vector3(0, 2, 0);
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

        [Test]
        public void TestBodyPosesEnabledDictionary()
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

            // Expect the root body disabled and the attached one enabled.
            Assert.IsFalse(poseExtractor.IsPoseEnabled(0));
            Assert.IsTrue(poseExtractor.IsPoseEnabled(1));
            var bodyPosesEnabled = poseExtractor.GetBodyPosesEnabled();
            Assert.IsFalse(bodyPosesEnabled[rb1]);
            Assert.IsTrue(bodyPosesEnabled[rb2]);

            // Swap the values
            bodyPosesEnabled[rb1] = true;
            bodyPosesEnabled[rb2] = false;

            var poseExtractor2 = new RigidBodyPoseExtractor(rb1, null, null, bodyPosesEnabled);
            Assert.IsTrue(poseExtractor2.IsPoseEnabled(0));
            Assert.IsFalse(poseExtractor2.IsPoseEnabled(1));
        }
    }
}
