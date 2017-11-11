using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BendScript : MonoBehaviour {

    public float CurrentBend = 0f;
    public float DesiredBend = 0f;
    public Vector3 BendMask = new Vector3(1, 0, 0);

    public void Reset()
    {
        CurrentBend = 0;
        DesiredBend = 0;
        transform.rotation = Quaternion.identity;
    }
}
