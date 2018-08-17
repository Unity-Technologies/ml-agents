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
			bp.vectorActionSpaceType = SpaceType.continuous;
			bp.vectorActionSize = new int[1] {4};
			var masker = new ActionMasker(bp);
			Assert.Catch<UnityAgentsException>(() => masker.ModifyActionMask(0,new int[1]{0}));
			masker.GetMask();
		}

		[Test]
		public void NullMask()
		{
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			var masker = new ActionMasker(bp);
			var mask = masker.GetMask();
			Assert.IsNull(mask);
		}
		
		[Test]
		public void FirstBranchMask()
		{
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			bp.vectorActionSize = new int[3] {4, 5, 6};
			var masker = new ActionMasker(bp);
			var mask = masker.GetMask();
			Assert.IsNull(mask);
			masker.ModifyActionMask(0,new int[]{1,2,3});
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
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			bp.vectorActionSize = new int[3] {4, 5, 6};
			var masker = new ActionMasker(bp);
			bool[] mask = masker.GetMask();
			masker.ModifyActionMask(1, new int[]{1,2,3});
			mask = masker.GetMask();
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
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			bp.vectorActionSize = new int[3] {4, 5, 6};
			var masker = new ActionMasker(bp);
			var mask = masker.GetMask();
			masker.ModifyActionMask(1, new int[]{1,2,3});
			mask = masker.GetMask();
			masker.ResetMask();
			mask = masker.GetMask();
			for (var i = 0; i < 15; i++)
			{
				Assert.IsFalse(mask[i]);
			}
		}

		[Test]
		public void ThrowsError()
		{
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			bp.vectorActionSize = new int[3] {4, 5, 6};
			var masker = new ActionMasker(bp);
			
			Assert.Catch<UnityAgentsException>(
				() => masker.ModifyActionMask(0, new int[1]{5}));
			Assert.Catch<UnityAgentsException>(
				() => masker.ModifyActionMask(1, new int[1]{5}));
			masker.ModifyActionMask(2, new int[1] {5});
			Assert.Catch<UnityAgentsException>(
				() => masker.ModifyActionMask(3, new int[1]{1}));
			masker.GetMask();
			masker.ResetMask();
			masker.ModifyActionMask(0, new int[4] {0, 1, 2, 3});	
			Assert.Catch<UnityAgentsException>(
				() => masker.GetMask());
		}
		
		[Test]
		public void MultipleMaksEdit()
		{
			var bp = new BrainParameters();
			bp.vectorActionSpaceType = SpaceType.discrete;
			bp.vectorActionSize = new int[3] {4, 5, 6};
			var masker = new ActionMasker(bp);
			masker.ModifyActionMask(0, new int[2] {0, 1});
			masker.ModifyActionMask(0, new int[1] {3});
			masker.ModifyActionMask(2, new int[1] {1});
			var mask = masker.GetMask();
			for (var i = 0; i < 15; i++)
			{
				if ((i == 0) || (i == 1) || (i == 3)|| (i == 10))
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