//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class PushAgentDiverse : PushAgentBasic
{

    VectorSensorComponent m_DiversitySettingSensor;
    public int m_DiversitySetting = 0;
    public int m_NumDiversityBehaviors = 3;

    public override void Initialize()
    {
        base.Initialize();
        GetComponent<VectorSensorComponent>().CreateSensors();
        m_DiversitySettingSensor = GetComponent<VectorSensorComponent>();
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);
        m_DiversitySettingSensor.GetSensor().Reset();
        m_DiversitySettingSensor.GetSensor().AddOneHotObservation(m_DiversitySetting, m_NumDiversityBehaviors);
    }

    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        m_DiversitySetting = Random.Range(0, m_NumDiversityBehaviors);
    }
}
