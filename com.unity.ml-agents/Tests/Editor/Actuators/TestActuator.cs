using Unity.MLAgents.Actuators;
namespace Unity.MLAgents.Tests.Actuators
{
    internal class TestActuator : IActuator
    {
        public ActionBuffers LastActionBuffer;
        public int[][] Masks;
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
            for (var i = 0; i < Masks.Length; i++)
            {
                actionMask.WriteMask(i, Masks[i]);
            }
        }

        public int TotalNumberOfActions { get; }
        public ActionSpaceDef ActionSpaceDef { get; }

        public string Name { get; }

        public void ResetData()
        {
        }
    }
}
