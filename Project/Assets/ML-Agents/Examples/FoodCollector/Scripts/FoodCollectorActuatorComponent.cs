using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class FoodCollectorActuatorComponent : ActuatorComponent
{
    public FoodCollectorController foodCollectorController;
    ActionSpec m_ActionSpec = ActionSpec.MakeDiscrete(2);

    /// <summary>
    /// Creates a FoodCollectorActuator.
    /// </summary>
    /// <returns></returns>
    public override IActuator CreateActuator()
    {
        return new FoodCollectorActuator(foodCollectorController);
    }

    public override ActionSpec ActionSpec
    {
        get { return m_ActionSpec; }
    }
}

/// <summary>
/// Actuator that handles the shoot action in FoodCollector
/// </summary>
public class FoodCollectorActuator : IActuator
{
    public FoodCollectorController foodCollectorController;
    ActionSpec m_ActionSpec;

    public FoodCollectorActuator(FoodCollectorController agent)
    {
        foodCollectorController = agent;
        m_ActionSpec = ActionSpec.MakeDiscrete(2);
    }

    public ActionSpec ActionSpec
    {
        get { return m_ActionSpec; }
    }

    /// <inheritdoc/>
    public String Name
    {
        get { return "Food"; }
    }

    public void ResetData()
    {

    }

    public void OnActionReceived(ActionBuffers actionBuffers)
    {
        foodCollectorController.Shoot(actionBuffers.DiscreteActions);
    }

    public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {

    }

}


