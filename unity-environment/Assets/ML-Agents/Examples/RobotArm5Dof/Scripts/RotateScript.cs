using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateScript : MonoBehaviour {

    public float CurrentRotation = 0f;
    public float DesiredRotation = 0f;

    public void Reset()
    {
        CurrentRotation = 180f;
        DesiredRotation = 180f;
        transform.rotation = Quaternion.identity;
        transform.Rotate(0, 180f, 0);
    }
}
