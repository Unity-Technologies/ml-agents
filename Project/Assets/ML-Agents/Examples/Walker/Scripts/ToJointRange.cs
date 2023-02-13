using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ToJointRange : MonoBehaviour
{

    ConfigurableJoint joint;
    float lowX;
    float lowY;
    float lowZ;
    float highX;
    float highY;
    float highZ;
        
    // Start is called before the first frame update
    void Initialize()
    {
        joint = this.GetComponent<ConfigurableJoint>();
        joint.angularXMotion = UnityEngine.ConfigurableJointMotion.Limited;
        joint.angularYMotion = UnityEngine.ConfigurableJointMotion.Limited;
        joint.angularZMotion = UnityEngine.ConfigurableJointMotion.Limited;   
        
        lowX = joint.lowAngularXLimit.limit;     
        //lowY = joint.lowAngularYLimit.limit;     
        //lowZ = joint.lowAngularZLimit.limit;     

        highX = joint.highAngularXLimit.limit;     
        highY = joint.angularYLimit.limit;      
        highZ = joint.angularZLimit.limit;            
    }

    // Update is called once per frame
    void Update()
    {

    }
}
