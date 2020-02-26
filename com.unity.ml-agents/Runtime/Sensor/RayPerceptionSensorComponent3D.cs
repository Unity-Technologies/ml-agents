using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A component for 3D Ray Perception.
    /// </summary>
    [AddComponentMenu("ML Agents/Ray Perception Sensor 3D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent3D : RayPerceptionSensorComponentBase
    {
        /// <summary>
        /// Ray start is offset up or down by this amount.
        /// </summary>
        [Header("3D Properties", order = 100)]
        [Range(-10f, 10f)]
        [Tooltip("Ray start is offset up or down by this amount.")]
        public float startVerticalOffset;

        /// <summary>
        /// Ray end is offset up or down by this amount.
        /// </summary>
        [Range(-10f, 10f)]
        [Tooltip("Ray end is offset up or down by this amount.")]
        public float endVerticalOffset;

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
