// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using Unity.MLAgents.Sensors;
// using System;
// using System.Linq;


// /// <summary>
// /// A simple example of a SensorComponent.
// /// This should be added to the same GameObject as the BasicController
// /// </summary>
// public class AttentionSensorComponent : SensorComponent
// {

//     public int ObservableSize;
//     public int MaxNumObservables;

//     /// <summary>
//     /// Creates a BasicSensor.
//     /// </summary>
//     /// <returns></returns>
//     public override ISensor CreateSensor()
//     {
//         return new AttentionSensor(transform, ObservableSize, MaxNumObservables);
//     }

//     /// <inheritdoc/>
//     public override int[] GetObservationShape()
//     {
//         return new[] { MaxNumObservables, ObservableSize, 1};
//     }
// }

// /// <summary>
// /// Simple Sensor implementation that uses a one-hot encoding of the Agent's
// /// position as the observation.
// /// </summary>
// public class AttentionSensor : ISensor
// {
//     int m_ObservableSize;
//     int m_MaxNumObservables;
//     float[] m_ObservationBuffer;
//     int m_CurrentNumObservables;
//     Transform m_AgentTransform;

//     public AttentionSensor(Transform AgentTransform, int ObservableSize, int MaxNumObservables)
//     {
//         m_ObservableSize = ObservableSize;
//         m_MaxNumObservables = MaxNumObservables;
//         m_AgentTransform = AgentTransform;
//         m_ObservationBuffer = new float[m_ObservableSize * m_MaxNumObservables];
//         m_CurrentNumObservables = 0;
//     }

//     /// <summary>
//     /// Generate the observations for the sensor.
//     /// In this case, the observations are all 0 except for a 1 at the position of the agent.
//     /// </summary>
//     /// <param name="output"></param>
//     public int Write(ObservationWriter writer)
//     {
//         for (int i = 0; i < m_ObservableSize * m_MaxNumObservables; i++){
//             writer[i] = m_ObservationBuffer[i];
//         }
//         return m_ObservableSize * m_MaxNumObservables;
//     }

//     public byte[] GetCompressedObservation()
//     {
//         return new byte[0];
//     }

//     public int[] GetObservationShape()
//     {
//         return new[] { m_MaxNumObservables, m_ObservableSize,1 };
//     }

//     /// <inheritdoc/>
//     public void Update() {
//         Reset();
//         var bullets = m_AgentTransform.parent.GetComponentsInChildren<Bullet>();
//         // Sort by closest :
//         Array.Sort(bullets , (a, b) => Vector3.Distance(a.transform.position, m_AgentTransform.position) - Vector3.Distance(b.transform.position, m_AgentTransform.position) > 0 ? 1 : -1);

//         // foreach (Bullet b in bullets)
//         // {
//         //     b.transform.localScale = 0.5f * new Vector3(1,1,1);
//         // }



//         foreach (Bullet b in bullets)
//         {
//             if (m_CurrentNumObservables >= m_MaxNumObservables){
//                 break;
//             }
//             m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 0] = (b.transform.position.x - m_AgentTransform.parent.position.x) / 10f;
//             m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 1] = (b.transform.position.z - m_AgentTransform.parent.position.z) / 10f;
//             //m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 0] = (b.transform.position.x - m_AgentTransform.position.x) / 10f;
//             //m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 1] = (b.transform.position.z - m_AgentTransform.position.z) / 10f;
//             m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 2] = b.transform.forward.x;
//             m_ObservationBuffer[m_CurrentNumObservables * m_ObservableSize + 3] = b.transform.forward.z;
//             m_CurrentNumObservables += 1;
//             // b.transform.localScale = 1f* new Vector3(1,1,1);
//         }

//     }

//     /// <inheritdoc/>
//     public void Reset() {
//         m_CurrentNumObservables = 0;
//         Array.Clear(m_ObservationBuffer, 0, m_ObservationBuffer.Length);
//     }

//     public SensorCompressionType GetCompressionType()
//         {
//             return SensorCompressionType.None;
//         }

//     /// <summary>
//     /// Accessor for the name of the sensor.
//     /// </summary>
//     /// <returns>Sensor name.</returns>
//     public string GetName()
//     {
//         return "AttentionSensor";
//     }

// }
