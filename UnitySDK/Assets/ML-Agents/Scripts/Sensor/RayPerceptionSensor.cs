using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RayPerceptionSensor : ISensor
    {
        float[] m_Observations;
        int[] m_Shape;
        string m_Name;

        float m_RayDistance;
        List<string> m_DetectableObjects;
        float[] m_Angles;

        float m_StartOffset = 0.0f;
        float m_EndOffset = 0.0f;
        float m_CastRadius = 0.5f;
        Transform m_Transform;

        public RayPerceptionSensor(string name, float rayDistance, List<string> detectableObjects, float[] angles,
            Transform transform, float startOffset, float endOffset, float castRadius)
        {
            var numObservations = (detectableObjects.Count + 2) * angles.Length;
            m_Shape = new[] { numObservations };
            m_Name = name;

            m_Observations = new float[numObservations];

            m_RayDistance = rayDistance;
            m_DetectableObjects = detectableObjects;
            m_Angles = angles;
            m_Transform = transform;
        }

        public int Write(WriteAdapter adapter)
        {
            DoRaycasts();
            adapter.AddRange(m_Observations);
            return m_Observations.Length;
        }

        void DoRaycasts()
        {
            RayPerception3D.PerceiveStatic(
                m_RayDistance, m_Angles, m_DetectableObjects, m_StartOffset, m_EndOffset,
                m_CastRadius, m_Transform, m_Observations
            );
            // TODO 2D case too

        }

        public void Update()
        {
        }

        public int[] GetFloatObservationShape()
        {
            return m_Shape;
        }

        public string GetName()
        {
            return m_Name;
        }

        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        public virtual SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }
    }
}
