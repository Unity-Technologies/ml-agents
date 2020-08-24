using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteAlways]
public class QuaternionDebug : MonoBehaviour
{
    public Transform staticObj;
    public Transform dynamicObj;

    public float rotDotProd;
    public float angleDiff;
    public float normalizedAngleDiff;
    public Quaternion fromToRot;
    public Vector3 fromToRotVector;

    public Vector3 inverseDiffRotMethod;

    public Vector3 crossProdDynamicCube;
    public float xAngleDiff;
    public float yAngleDiff;
    public float zAngleDiff;
    public float dotOfCrossProdDiff;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        var s = staticObj.rotation;
        var d = dynamicObj.rotation;
        rotDotProd = Quaternion.Dot(s, d);
        fromToRot = Quaternion.FromToRotation(staticObj.forward, dynamicObj.forward);
        fromToRotVector = fromToRot.eulerAngles;
        angleDiff = Quaternion.Angle(s, d);
        normalizedAngleDiff = 1 - (angleDiff / 180);
        inverseDiffRotMethod = (s * Quaternion.Inverse(d)).eulerAngles;
        crossProdDynamicCube = Vector3.Cross(dynamicObj.up, dynamicObj.forward);
        dotOfCrossProdDiff = Vector3.Dot(crossProdDynamicCube, staticObj.right);
//        xAngleDiff = Vector3.SignedAngle(dynamicObj.forward, staticObj.forward, dynamicObj.right);
        xAngleDiff = Vector3.SignedAngle(dynamicObj.forward, staticObj.forward, Vector3.up);
        yAngleDiff = Vector3.SignedAngle(dynamicObj.forward, staticObj.forward, dynamicObj.up);
        zAngleDiff = Vector3.SignedAngle(dynamicObj.forward, staticObj.forward, dynamicObj.forward);
    }
}
