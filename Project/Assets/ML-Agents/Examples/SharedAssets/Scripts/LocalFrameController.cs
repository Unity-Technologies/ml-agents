using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LocalFrameController : MonoBehaviour
{
    public void UpdateLocalFrame(Transform root)
    {
        var direction = root.forward;
        direction.y = 0;
        var lookRot = direction == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(direction);
        transform.SetPositionAndRotation(root.position, lookRot);
    }
}
