using System.Collections;
using System.Collections.Generic;
using InspectorGadgets;
using UnityEngine;

public class LocalFrameController : MonoBehaviour
{
    public void UpdateLocalFrame(Transform root)
    {
        var heading = CalculateHeading(root);
        var lookRot = Quaternion.AngleAxis(heading, Vector3.up);
        transform.SetPositionAndRotation(root.position, lookRot);
    }

    float CalculateHeading(Transform root)
    {
        var rotationDirection = Quaternion.FromToRotation(root.forward, Vector3.forward);
        var heading = -rotationDirection.eulerAngles.y;
        return heading;
    }
}
