using Unity.MLAgents.Sensors;


public class GoalSensorComponent : SensorComponent
{
    public int observationSize;
    public GoalSensor goalSensor;
    /// <summary>
    /// Creates a GoalSensor.
    /// </summary>
    /// <returns></returns>
    public override ISensor CreateSensor()
    {
        goalSensor = new GoalSensor(observationSize);
        return goalSensor;
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        return new[] { observationSize };
    }

    public void AddGoal(float goal)
    {
        if(goalSensor != null)
        {
            goalSensor.AddObservation(goal);
        }
    }
}

public class GoalSensor : VectorSensor
{

    public GoalSensor(int observationSize, string name = null) : base(observationSize)
    {
        if (name == null)
        {
            name = $"GoalSensor_size{observationSize}";
        }
    }

    public override SensorType GetSensorType()
    {
        return SensorType.Goal;
    }
}
