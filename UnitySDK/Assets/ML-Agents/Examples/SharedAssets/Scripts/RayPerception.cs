using System;
using System.Collections.Generic;
using UnityEngine;

[Obsolete]
public abstract class RayPerception : MonoBehaviour
{
    protected float[] m_PerceptionBuffer;

    abstract public IList<float> Perceive(float rayDistance,
        float[] rayAngles, string[] detectableObjects,
        float startOffset=0.0f, float endOffset=0.0f);


}
