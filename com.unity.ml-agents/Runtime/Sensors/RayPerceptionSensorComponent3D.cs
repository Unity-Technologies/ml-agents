using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents.Sensors
{
    /// <summary>
    /// A component for 3D Ray Perception.
    /// </summary>
    [AddComponentMenu("ML Agents/Ray Perception Sensor 3D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent3D : RayPerceptionSensorComponentBase
    {
        [HideInInspector, SerializeField, FormerlySerializedAs("startVerticalOffset")]
        [Range(-10f, 10f)]
        [Tooltip("Ray start is offset up or down by this amount.")]
        float m_StartVerticalOffset;

        /// <summary>
        /// Ray start is offset up or down by this amount.
        /// </summary>
        public float startVerticalOffset
        {
            get => m_StartVerticalOffset;
            set { m_StartVerticalOffset = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("endVerticalOffset")]
        [Range(-10f, 10f)]
        [Tooltip("Ray end is offset up or down by this amount.")]
        float m_EndVerticalOffset;

        /// <summary>
        /// Ray end is offset up or down by this amount.
        /// </summary>
        public float endVerticalOffset
        {
            get => m_EndVerticalOffset;
            set { m_EndVerticalOffset = value; UpdateSensor(); }
        }

        /// <inheritdoc/>
        public override RayPerceptionCastType GetCastType()
        {
            return RayPerceptionCastType.Cast3D;
        }

        /// <inheritdoc/>
        public override float GetStartVerticalOffset()
        {
            return startVerticalOffset;
        }

        /// <inheritdoc/>
        public override float GetEndVerticalOffset()
        {
            return endVerticalOffset;
        }
    }
}
