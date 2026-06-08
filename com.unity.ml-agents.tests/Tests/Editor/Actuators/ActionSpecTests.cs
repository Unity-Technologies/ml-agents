using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Tests.Actuators
{
    [TestFixture]
    public class ActionSpecTests
    {
        [Test]
        public void ActionSpecCombineTest()
        {
            var as0 = new ActionSpec(3, new[] { 3, 2, 1 });
            var as1 = new ActionSpec(1, new[] { 35, 122, 1, 3, 8, 3 });

            var as0NumCon = 3;
            var as0NumDis = as0.NumDiscreteActions;
            var as1NumCon = 1;
            var as1NumDis = as1.NumDiscreteActions;
            var branchSizes = new List<int>();
            branchSizes.AddRange(as0.BranchSizes);
            branchSizes.AddRange(as1.BranchSizes);

            var asc = ActionSpec.Combine(as0, as1);

            Assert.AreEqual(as0NumCon + as1NumCon, asc.NumContinuousActions);
            Assert.AreEqual(as0NumDis + as1NumDis, asc.NumDiscreteActions);
            Assert.IsTrue(branchSizes.ToArray().SequenceEqual(asc.BranchSizes));

            as0 = new ActionSpec(3);
            as1 = new ActionSpec(1);
            asc = ActionSpec.Combine(as0, as1);
            Assert.IsEmpty(asc.BranchSizes);
        }
    }
}
