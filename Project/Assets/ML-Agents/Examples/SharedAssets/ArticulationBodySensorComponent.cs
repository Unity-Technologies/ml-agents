using Unity.MLAgents.Sensors;
using UnityEngine;
public class ArticulationBodySensorComponent : SensorComponent
{
    public ArticulationBody RootBody;

    [SerializeField]
    public ArticulationBodySensorSettings Settings = new ArticulationBodySensorSettings();

    /// <inheritdoc/>
    public override ISensor CreateSensor()
    {
        return new ArticulationBodySensor(RootBody, Settings);
    }

    /// <inheritdoc/>
    public override int[] GetObservationShape()
    {
        return new[] { ArticulationBodySensor.GetArticulationSensorSize(RootBody, Settings) };
    }
}
