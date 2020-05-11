using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[ExecuteAlways]
public class MeshSkewFix : MonoBehaviour
{
    public bool fix;
    public GameObject rootGameObject;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (fix)
        {
            fix = false;
            foreach (var t in GetComponentsInChildren<Transform>())
            {
                var joint = t.GetComponent<ConfigurableJoint>();
                if (joint)
                {
                    var meshFilter = t.GetComponent<MeshFilter>();
                    var meshRend = t.GetComponent<MeshFilter>();
                }
                
            }
        }
        
    }
}
