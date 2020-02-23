using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    [AddComponentMenu("ML Agents/Ray Perception Sensor 3D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent3D : RayPerceptionSensorComponentBase
    {
        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("startVerticalOffset")]
        [Range(-10f, 10f)]
        [Tooltip("Ray start is offset up or down by this amount.")]
        float m_StartVerticalOffset;
        public float startVerticalOffset
        {
            get => m_StartVerticalOffset;
            set => m_StartVerticalOffset = value;
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("endVerticalOffset")]
        [Range(-10f, 10f)]
        [Tooltip("Ray end is offset up or down by this amount.")]
        float m_EndVerticalOffset;
        public float endVerticalOffset
        {
            get => m_EndVerticalOffset;
            set => m_EndVerticalOffset = value;
        }

        public override RayPerceptionCastType GetCastType()
        {
            return RayPerceptionCastType.Cast3D;
        }

        public override float GetStartVerticalOffset()
        {
            return startVerticalOffset;
        }

        public override float GetEndVerticalOffset()
        {
            return endVerticalOffset;
        }
    }
}
