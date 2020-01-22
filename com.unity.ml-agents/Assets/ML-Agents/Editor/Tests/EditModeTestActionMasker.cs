using NUnit.Framework;

namespace MLAgents.Tests
{
    public class EditModeTestActionMasker
    {
        [Test]
        public void Contruction()
        {
            var bp = new BrainParameters();
            var masker = new ActionMasker(bp);
            Assert.IsNotNull(masker);
        }

        [Test]
        public void FailsWithContinuous()
        {
            var bp = new BrainParameters();
            bp.vectorActionSpaceType = SpaceType.Continuous;
            bp.vectorActionSize = new[] {4};
            var masker = new ActionMasker(bp);
            masker.SetActionMask(0, new[] {0});
            Assert.Catch<UnityAgentsException>(() => masker.GetMask());
        }

        [Test]
        public void NullMask()
        {
            var bp = new BrainParameters();
            bp.vectorActionSpaceType = SpaceType.Discrete;
            var masker = new ActionMasker(bp);
            var mask = masker.GetMask();
            Assert.IsNull(mask);
        }

        [Test]
        public void FirstBranchMask()
        {
            var bp = new BrainParameters();
            bp.vectorActionSpaceType = SpaceType.Discrete;
            bp.vectorActionSize = new[] {4, 5, 6};
            var masker = new ActionMasker(bp);
            var mask = masker.GetMask();
            Assert.IsNull(mask);
            masker.SetActionMask(0, new[] {1, 2, 3});
            mask = masker.GetMask();
            Assert.IsFalse(mask[0]);
            Assert.IsTrue(mask[1]);
            Assert.IsTrue(mask[2]);
            Assert.IsTrue(mask[3]);
            Assert.IsFalse(mask[4]);
            Assert.AreEqual(mask.Length, 15);
        }

        [Test]
        public void SecondBranchMask()
        {
            var bp = new BrainParameters
            {
                vectorActionSpaceType = SpaceType.Discrete,
                vectorActionSize = new[] { 4, 5, 6 }
            };
            var masker = new ActionMasker(bp);
            masker.SetActionMask(1, new[] {1, 2, 3});
            var mask = masker.GetMask();
            Assert.IsFalse(mask[0]);
            Assert.IsFalse(mask[4]);
            Assert.IsTrue(mask[5]);
            Assert.IsTrue(mask[6]);
            Assert.IsTrue(mask[7]);
            Assert.IsFalse(mask[8]);
            Assert.IsFalse(mask[9]);
        }

        [Test]
        public void MaskReset()
        {
            var bp = new BrainParameters
            {
                vectorActionSpaceType = SpaceType.Discrete,
                vectorActionSize = new[] { 4, 5, 6 }
            };
            var masker = new ActionMasker(bp);
            masker.SetActionMask(1, new[] {1, 2, 3});
            masker.ResetMask();
            var mask = masker.GetMask();
            for (var i = 0; i < 15; i++)
            {
                Assert.IsFalse(mask[i]);
            }
        }

        [Test]
        public void ThrowsError()
        {
            var bp = new BrainParameters
            {
                vectorActionSpaceType = SpaceType.Discrete,
                vectorActionSize = new[] { 4, 5, 6 }
            };
            var masker = new ActionMasker(bp);

            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionMask(0, new[] {5}));
            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionMask(1, new[] {5}));
            masker.SetActionMask(2, new[] {5});
            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionMask(3, new[] {1}));
            masker.GetMask();
            masker.ResetMask();
            masker.SetActionMask(0, new[] {0, 1, 2, 3});
            Assert.Catch<UnityAgentsException>(
                () => masker.GetMask());
        }

        [Test]
        public void MultipleMaskEdit()
        {
            var bp = new BrainParameters();
            bp.vectorActionSpaceType = SpaceType.Discrete;
            bp.vectorActionSize = new[] {4, 5, 6};
            var masker = new ActionMasker(bp);
            masker.SetActionMask(0, new[] {0, 1});
            masker.SetActionMask(0, new[] {3});
            masker.SetActionMask(2, new[] {1});
            var mask = masker.GetMask();
            for (var i = 0; i < 15; i++)
            {
                if ((i == 0) || (i == 1) || (i == 3) || (i == 10))
                {
                    Assert.IsTrue(mask[i]);
                }
                else
                {
                    Assert.IsFalse(mask[i]);
                }
            }
        }
    }
}
