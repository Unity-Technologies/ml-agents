using Unity.MLAgents.Actuators;
namespace Unity.MLAgents.Tests.Actuators
{
    internal class TestActuator : IActuator
    {
        public ActionBuffers LastActionBuffer;
        public int[][] Masks;
        public bool m_HeuristicCalled;
        public int m_DiscreteBufferSize;

        public TestActuator(ActionSpec actuatorSpace, string name)
        {
            ActionSpec = actuatorSpace;

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
                foreach (var actionIndex in Masks[i])
                {
                    actionMask.SetActionEnabled(i, actionIndex, false);
                }
            }
        }

        public ActionSpec ActionSpec { get; }

        public string Name { get; }

        public void ResetData()
        {
        }

        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            m_HeuristicCalled = true;
            m_DiscreteBufferSize = actionBuffersOut.DiscreteActions.Length;
        }
    }
}
