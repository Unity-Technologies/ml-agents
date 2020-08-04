using NUnit.Framework;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents.Tests.Actuators
{
    internal class TestActuator : IActuator
    {
        public ActionBuffers LastActionBuffer;
        //public int branch;
        //public int[] MaskIndexes;
        public TestActuator(ActionSpaceDef actuatorSpace, string name)
        {
            ActionSpaceDef = actuatorSpace;
            TotalNumberOfActions = actuatorSpace.NumContinuousActions +
                actuatorSpace.NumDiscreteActions;
            Name = name;
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            LastActionBuffer = actionBuffers;
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            // if (MaskIndexes != null)
            // {
            //     actionMask.WriteMask(branch, MaskIndexes);
            // }
        }

        public int TotalNumberOfActions { get; }
        public ActionSpaceDef ActionSpaceDef { get; }

        public string Name { get; }

        public void ResetData()
        {
        }
    }
    [TestFixture]
    public class ActuatorManagerTests
    {

    }
}
