using System;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// </summary>
    public class BasicActuatorComponent : ActuatorComponent
    {
        public BasicController basicController;

        /// <summary>
        /// Creates a BasicActuator.
        /// </summary>
        /// <returns></returns>
        public override IActuator CreateActuator()
        {
            return new BasicActuator(basicController);
        }
        // TOOD action spec
    }

    /// <summary>
    /// </summary>
    public class BasicActuator : IActuator
    {
        public BasicController basicController;
        ActionSpec m_ActionSpec;

        public BasicActuator(BasicController controller)
        {
            basicController = controller;
            // TODO add params version of MakeDiscrete?
            m_ActionSpec = ActionSpec.MakeDiscrete(new[] {3} );
        }

        // TODO do we even need this?
        public int TotalNumberOfActions
        {
            get { return m_ActionSpec.NumContinuousActions + m_ActionSpec.NumDiscreteActions; }
        }

        public ActionSpec ActionSpec
        {
            get { return m_ActionSpec; }
        }

        /// <inheritdoc/>
        public String Name
        {
            get { return "Basic"; }
        }

        public void ResetData()
        {

        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            var movement = actionBuffers.DiscreteActions[0];

            var direction = 0;

            switch (movement)
            {
                case 1:
                    direction = -1;
                    break;
                case 2:
                    direction = 1;
                    break;
            }

            basicController.MoveDirection(direction);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {

        }

    }
}
