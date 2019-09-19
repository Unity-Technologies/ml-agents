using System.Collections.Generic;
using UnityEngine;

public abstract class RayPerception : MonoBehaviour
{
    protected List<float> m_PerceptionBuffer = new List<float>();

    public virtual List<float> Perceive(float rayDistance,
        float[] rayAngles, string[] detectableObjects,
        float startOffset, float endOffset)
    {
        return m_PerceptionBuffer;
    }

    /// <summary>
    /// Converts degrees to radians.
    /// </summary>
    public static float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }
}
