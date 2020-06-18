using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Tests.Sensors
{
    public class HierarchyUtilTests
    {
        class UselessHierarchyUtil : HierarchyUtil
        {
            protected override QTTransform GetTransformAt(int index)
            {
                return QTTransform.Identity;
            }

            public void Init(int[] parentIndices)
            {
                SetParentIndices(parentIndices);
            }
        }

        [Test]
        public void TestEmptyUtil()
        {
            var hierarchyUtil = new UselessHierarchyUtil();

            // These should be no-ops
            hierarchyUtil.UpdateLocalSpaceTransforms();
            hierarchyUtil.UpdateModelSpaceTransforms();

            Assert.AreEqual(0, hierarchyUtil.NumTransforms);
        }

        [Test]
        public void TestSimpleUtil()
        {
            var hierarchyUtil = new UselessHierarchyUtil();
            var parentIndices = new[] { -1, 0 };
            hierarchyUtil.Init(parentIndices);
            Assert.AreEqual(2, hierarchyUtil.NumTransforms);
        }


        /// <summary>
        /// A simple "chain" hierarchy, where each object is parented to the one before it.
        ///   0 <- 1 <- 2 <- ...
        /// </summary>
        class ChainHierarchyUtil : HierarchyUtil
        {
            public Vector3 offset;
            public ChainHierarchyUtil(int size)
            {
                var parents = new int[size];
                for (var i = 0; i < size; i++)
                {
                    parents[i] = i - 1;
                }
                SetParentIndices(parents);
            }

            protected override QTTransform GetTransformAt(int index)
            {
                var rotation = Quaternion.identity;
                var translation = offset + new Vector3(index, index, index);
                return new QTTransform
                {
                    Rotation = rotation,
                    Translation = translation
                };
            }
        }

        [Test]
        public void TestChain()
        {
            var size = 4;
            var chain = new ChainHierarchyUtil(size);
            chain.offset = new Vector3(.5f, .75f, .333f);

            chain.UpdateModelSpaceTransforms();
            chain.UpdateLocalSpaceTransforms();

            // Root transforms are currently always the identity.
            Assert.IsTrue(chain.ModelSpacePose[0] == QTTransform.Identity);
            Assert.IsTrue(chain.LocalSpacePose[0] == QTTransform.Identity);

            // Check the non-root transforms
            for (var i = 1; i < size; i++)
            {
                var modelSpace = chain.ModelSpacePose[i];
                var expectedModelTranslation = new Vector3(i, i, i);
                Assert.IsTrue(expectedModelTranslation == modelSpace.Translation);

                var localSpace = chain.LocalSpacePose[i];
                var expectedLocalTranslation = new Vector3(1, 1, 1);
                Assert.IsTrue(expectedLocalTranslation == localSpace.Translation);
            }
        }

    }
}
