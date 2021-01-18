using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SequenceTile : MonoBehaviour
{
    public int NumberValue;
    [HideInInspector]
    public bool visited = false;

    // [HideInInspector]
    public MeshRenderer rend;
    // Start is called before the first frame update
    void Awake()
    {
        rend = GetComponentInChildren<MeshRenderer>();
    }

    // Update is called once per frame
    void Update()
    {

    }
}
