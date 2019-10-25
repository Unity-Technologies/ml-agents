using System.Collections.Generic;
using UnityEngine;

public abstract class RayPerception : MonoBehaviour
{
    protected List<float> m_PerceptionBuffer = new List<float>();

    abstract public List<float> Perceive(float rayDistance,
        float[] rayAngles, string[] detectableObjects,
        float startOffset=0.0f, float endOffset=0.0f);

    /// <summary>
    /// Converts degrees to radians.
    /// </summary>
    public static float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }
}
