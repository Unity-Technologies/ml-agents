using Unity.MLAgents.Sensors;


public class TaskSensorComponent : SensorComponent
{
    public int observationSize;
    public TaskSensor task_sensor;
    /// <summary>
    /// Creates a TaskSensor.
    /// </summary>
    /// <returns></returns>
    public override ISensor CreateSensor()
    {
        task_sensor = new TaskSensor(observationSize);
        return task_sensor;
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        return new[] { observationSize };
    }

    public void AddParameterizaton(float parameter)
    {
        if(task_sensor != null)
        {
            task_sensor.AddObservation(parameter);
        }
    }
}

public class TaskSensor : VectorSensor
{

    public TaskSensor(int observationSize, string name = null) : base(observationSize)
    {
        if (name == null)
        {
            name = $"TaskSensor_size{observationSize}";
        }
    }

    public override SensorType GetSensorType()
    {
        return SensorType.Parameterization;
    }
}
