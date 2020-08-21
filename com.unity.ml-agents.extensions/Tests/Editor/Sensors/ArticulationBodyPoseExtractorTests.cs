#if UNITY_2020_1_OR_NEWER
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class ArticulationBodyPoseExtractorTests
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
            var poseExtractor = new ArticulationBodyPoseExtractor(null);
            // These should be no-ops
            poseExtractor.UpdateLocalSpacePoses();
            poseExtractor.UpdateModelSpacePoses();

            Assert.AreEqual(0, poseExtractor.NumPoses);
        }

        [Test]
        public void TestSingleBody()
        {
            var go = new GameObject();
            var rootArticBody = go.AddComponent<ArticulationBody>();
            var poseExtractor = new ArticulationBodyPoseExtractor(rootArticBody);
            Assert.AreEqual(1, poseExtractor.NumPoses);
        }

        [Test]
        public void TestTwoBodies()
        {
            // * rootObj
            //   - rootArticBody
            //   * leafGameObj
            //     - leafArticBody
            var rootObj = new GameObject();
            var rootArticBody = rootObj.AddComponent<ArticulationBody>();

            var leafGameObj = new GameObject();
            var leafArticBody = leafGameObj.AddComponent<ArticulationBody>();
            leafGameObj.transform.SetParent(rootObj.transform);

            leafArticBody.jointType = ArticulationJointType.RevoluteJoint;

            var poseExtractor = new ArticulationBodyPoseExtractor(rootArticBody);
            Assert.AreEqual(2, poseExtractor.NumPoses);
            Assert.AreEqual(-1, poseExtractor.GetParentIndex(0));
            Assert.AreEqual(0, poseExtractor.GetParentIndex(1));
        }
    }
}
#endif
