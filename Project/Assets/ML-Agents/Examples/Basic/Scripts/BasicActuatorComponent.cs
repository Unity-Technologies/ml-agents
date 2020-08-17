using System;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// A simple example of a ActuatorComponent.
    /// This should be added to the same GameObject as the BasicController
    /// </summary>
    public class BasicActuatorComponent : ActuatorComponent
    {
        public BasicController basicController;
        ActionSpec m_ActionSpec = ActionSpec.MakeDiscrete(3);

        /// <summary>
        /// Creates a BasicActuator.
        /// </summary>
        /// <returns></returns>
        public override IActuator CreateActuator()
        {
            return new BasicActuator(basicController);
        }

        public override ActionSpec ActionSpec
        {
            get { return m_ActionSpec; }
        }
    }

    /// <summary>
    /// Simple actuator that converts the action into a {-1, 0, 1} direction
    /// </summary>
    public class BasicActuator : IActuator
    {
        public BasicController basicController;
        ActionSpec m_ActionSpec;

        public BasicActuator(BasicController controller)
        {
            basicController = controller;
            m_ActionSpec = ActionSpec.MakeDiscrete(3);
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
