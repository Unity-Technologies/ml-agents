using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateScript : MonoBehaviour {

    public float CurrentRotation = 0f;
    public float DesiredRotation = 0f;

    public void Reset()
    {
        CurrentRotation = 0;
        DesiredRotation = 0;
        transform.rotation = Quaternion.identity;
    }
}
