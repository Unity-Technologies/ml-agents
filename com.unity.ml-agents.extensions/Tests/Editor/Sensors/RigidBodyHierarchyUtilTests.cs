using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class RigidBodyHierarchyUtilTests
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
            var hierarchyUtil = new RigidBodyHierarchyUtil(null);
            // These should be no-ops
            hierarchyUtil.UpdateLocalSpaceTransforms();
            hierarchyUtil.UpdateModelSpaceTransforms();

            Assert.AreEqual(0, hierarchyUtil.NumTransforms);
        }

        [Test]
        public void TestSingleBody()
        {
            var go = new GameObject();
            var rootRb = go.AddComponent<Rigidbody>();
            var hierarchyUtil = new RigidBodyHierarchyUtil(rootRb);
            Assert.AreEqual(1, hierarchyUtil.NumTransforms);
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

            var hierarchyUtil = new RigidBodyHierarchyUtil(rb1);
            Assert.AreEqual(2, hierarchyUtil.NumTransforms);
        }
    }
}
