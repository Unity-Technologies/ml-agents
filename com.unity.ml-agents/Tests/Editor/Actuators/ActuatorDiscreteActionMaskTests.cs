using System.Collections.Generic;
using NUnit.Framework;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Tests.Actuators
{
    [TestFixture]
    public class ActuatorDiscreteActionMaskTests
    {
        [Test]
        public void Construction()
        {
            var masker = new ActuatorDiscreteActionMask(new List<IActuator>(), 0, 0);
            Assert.IsNotNull(masker);
        }

        [Test]
        public void NullMask()
        {
            var masker = new ActuatorDiscreteActionMask(new List<IActuator>(), 0, 0);
            var mask = masker.GetMask();
            Assert.IsNull(mask);
        }

        [Test]
        public void FirstBranchMask()
        {
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new IActuator[] { actuator1 }, 15, 3);
            var mask = masker.GetMask();
            Assert.IsNull(mask);
            masker.SetActionEnabled(0, 1, false);
            masker.SetActionEnabled(0, 2, false);
            masker.SetActionEnabled(0, 3, false);
            mask = masker.GetMask();
            Assert.IsFalse(mask[0]);
            Assert.IsTrue(mask[1]);
            Assert.IsTrue(mask[2]);
            Assert.IsTrue(mask[3]);
            Assert.IsFalse(mask[4]);
            Assert.AreEqual(mask.Length, 15);
        }

        [Test]
        public void CanOverwriteMask()
        {
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new IActuator[] { actuator1 }, 15, 3);
            masker.SetActionEnabled(0, 1, false);
            var mask = masker.GetMask();
            Assert.IsTrue(mask[1]);

            masker.SetActionEnabled(0, 1, true);
            Assert.IsFalse(mask[1]);
        }

        [Test]
        public void SecondBranchMask()
        {
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new[] { actuator1 }, 15, 3);
            masker.SetActionEnabled(1, 1, false);
            masker.SetActionEnabled(1, 2, false);
            masker.SetActionEnabled(1, 3, false);
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
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new IActuator[] { actuator1 }, 15, 3);
            masker.SetActionEnabled(1, 1, false);
            masker.SetActionEnabled(1, 2, false);
            masker.SetActionEnabled(1, 3, false);
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
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new IActuator[] { actuator1 }, 15, 3);
            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionEnabled(0, 5, false));
            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionEnabled(1, 5, false));
            masker.SetActionEnabled(2, 5, false);
            Assert.Catch<UnityAgentsException>(
                () => masker.SetActionEnabled(3, 1, false));
            masker.GetMask();
            masker.ResetMask();
            masker.SetActionEnabled(0, 0, false);
            masker.SetActionEnabled(0, 1, false);
            masker.SetActionEnabled(0, 2, false);
            masker.SetActionEnabled(0, 3, false);
            Assert.Catch<UnityAgentsException>(
                () => masker.GetMask());
        }

        [Test]
        public void MultipleMaskEdit()
        {
            var actuator1 = new TestActuator(ActionSpec.MakeDiscrete(new[] { 4, 5, 6 }), "actuator1");
            var masker = new ActuatorDiscreteActionMask(new IActuator[] { actuator1 }, 15, 3);
            masker.SetActionEnabled(0, 0, false);
            masker.SetActionEnabled(0, 1, false);
            masker.SetActionEnabled(0, 3, false);
            masker.SetActionEnabled(2, 1, false);
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
