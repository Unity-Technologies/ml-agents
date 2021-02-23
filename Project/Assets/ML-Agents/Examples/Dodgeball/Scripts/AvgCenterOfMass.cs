using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Used for visualizing the average center of mass of a ragdoll
/// </summary>
[DisallowMultipleComponent]
[ExecuteInEditMode]
public class AvgCenterOfMass : MonoBehaviour
{
    [System.Serializable]
    public class ShiftCom
    {
        public Rigidbody rb;
        public Vector3 shiftComAmount;
    }
    /// <summary>
    /// Enable to show a green spehere at the current center of mass.
    /// </summary>
    [Tooltip("Enable to show a green spehere at the current center of mass.")]
    public bool active;
    public bool showCOMGizmos = true;
    public Vector3 avgCOMWorldSpace;
    public Vector3 avgCOMVelocityWorldSpace;
    public Vector3 previousAvgCOM;
    public Color avgCOMColor = Color.green;
    public Color bodyPartCOMColor = Color.yellow;
    List<Rigidbody> rbList = new List<Rigidbody>();
    public float totalMass;
    [Tooltip("Visualize Relative Pos")]
    public bool showBPPosRelToBody;
    public bool useTransformPoint = true;
    public bool useTransformVector;
    public bool useTransformDir;
    public bool showRBPos;
    public bool showRelPosVectorOnly;
    public bool showInverseTransformPointUnscaledRelToBody;
    public bool showInverseTransformPointRelToBody;
    public bool showInverseTransformVectorRelToBody;
    public bool showInverseTransformDirRelToBody;
    public Transform body_T;
    [Tooltip("ShiftCom")] public bool updateShiftCom;
    public List<ShiftCom> shiftComList = new List<ShiftCom>();
    void OnEnable()
    {
        SetUpRigidbodies();
    }

    void SetUpRigidbodies()
    {
        rbList.Clear();
        totalMass = 0;
        foreach (var item in GetComponentsInChildren<Rigidbody>())
        {
            rbList.Add(item);
            totalMass += item.mass;
        }
        foreach (var item in shiftComList)
        {
            item.rb.centerOfMass = item.shiftComAmount;
        }
    }

    // void FixedUpdate()
    // {
    //     if(Application.isPlaying)
    //     {
    //         avgCOMWorldSpace = Vector3.zero;

    //         foreach(var item in rbList)
    //         {
    //             if (item)
    //             {
    //                 avgCOMWorldSpace += item.worldCenterOfMass;
    //             }
    //         }

    //         //DRAW AVG GIZMOS
    //         avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
    //     }
    // }

    public Vector3 GetCoMWorldSpace()
    {
        Vector3 CoM = Vector3.zero;
        avgCOMWorldSpace = Vector3.zero;
        float c = 0f;

        foreach (var item in rbList)
        {
            CoM += item.worldCenterOfMass * item.mass;
            c += item.mass;
        }
        avgCOMWorldSpace = CoM / c;
        avgCOMVelocityWorldSpace = (avgCOMWorldSpace - previousAvgCOM) / Time.fixedDeltaTime;
        //        Debug.DrawRay(avgCOMWorldSpace, avgCOMVelocityWorldSpace, Color.green,Time.fixedDeltaTime);
        //        Debug.DrawRay(avgCOMWorldSpace, Vector3.ProjectOnPlane( avgCOMVelocityWorldSpace, Vector3.up), Color.green,Time.fixedDeltaTime);

        previousAvgCOM = avgCOMWorldSpace;
        return avgCOMWorldSpace;
    }


    void FixedUpdate()
    {


        if (Application.isPlaying)
        {
            // avgCOMWorldSpace = Vector3.zero;
            // foreach(var item in rbList)
            // {
            //     if (item)
            //     {
            //         avgCOMWorldSpace += item.worldCenterOfMass;
            //     }
            // }
            // //DRAW AVG GIZMOS
            // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space

            //            if (active)
            //            {
            GetCoMWorldSpace();
            //            }


            //            Vector3 CoM = Vector3.zero;
            //            avgCOMWorldSpace = Vector3.zero;
            //            float c = 0f;
            //
            //            foreach(var item in rbList)
            //            {
            //                CoM += item.worldCenterOfMass * item.mass;
            //                c += item.mass;
            //            }
            //            avgCOMWorldSpace = CoM/c;
            //            avgCOMVelocityWorldSpace = previousAvgCOM - avgCOMWorldSpace;
            //            Debug.DrawRay(avgCOMWorldSpace, avgCOMVelocityWorldSpace, Color.green,Time.fixedDeltaTime);
            //
            //            previousAvgCOM = avgCOMWorldSpace;
            //            // CoM /= c;
            //

            if (showBPPosRelToBody)
            {
                var pos = body_T.position;
                Matrix4x4 bodyMatrix = body_T.localToWorldMatrix;
                // get position from the last column
                var bodyPos = new Vector3(bodyMatrix[0, 3], bodyMatrix[1, 3], bodyMatrix[2, 3]);
                Debug.DrawRay(bodyPos, Vector3.up, Color.yellow, Time.fixedDeltaTime);
                foreach (var rb in rbList)
                {
                    if (showRBPos)
                    {
                        Debug.DrawRay(rb.position, Vector3.up, Color.green, Time.fixedDeltaTime);
                    }
                    if (rb.transform != body_T)
                    {
                        if (showRelPosVectorOnly)
                        {
                            var relPosVector = rb.position - body_T.position;
                            //                            Debug.DrawRay(body_T.position + body_T.InverseTransformPoint(rb.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                            Debug.DrawRay(body_T.position + relPosVector, Vector3.up, Color.red, Time.fixedDeltaTime);
                            //                            Vector3 currentLocalPosRelToMatrix = bodyMatrix.inverse.MultiplyPoint(rb.position);
                            Vector3 currentLocalPosRelToMatrix = bodyMatrix.inverse.MultiplyVector(rb.position - bodyPos);

                            Debug.DrawRay(body_T.position + currentLocalPosRelToMatrix, Vector3.up, Color.green, Time.fixedDeltaTime);
                        }
                        if (showInverseTransformPointUnscaledRelToBody)
                        {
                            //                            Debug.DrawRay(body_T.position + body_T.InverseTransformPoint(rb.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                            //                            Debug.DrawRay(body_T.position + body_T.InverseTransformPointUnscaled(rb.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                            //                            Debug.DrawRay(body_T.position + body_T.InverseTransformPointUnscaled(rb.transform.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                        }
                        if (showInverseTransformPointRelToBody)
                        {
                            //                            Debug.DrawRay(body_T.position + body_T.InverseTransformPoint(rb.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                            Debug.DrawRay(body_T.position + body_T.InverseTransformPoint(rb.position), Vector3.up, Color.red, Time.fixedDeltaTime);
                        }
                        if (showInverseTransformDirRelToBody)
                        {
                            Debug.DrawRay(body_T.InverseTransformDirection(rb.position), Vector3.up, Color.red, Time.fixedDeltaTime);
                        }
                        if (showInverseTransformVectorRelToBody)
                        {
                            Debug.DrawRay(body_T.position + body_T.InverseTransformVector(rb.position - body_T.position), Vector3.up, Color.red, Time.fixedDeltaTime);
                        }
                        //                        var localPosRelToBody = body.InverseTransformPoint(rb.position);
                        //    Debug.DrawRay(body_T.position + body_T.InverseTransformPoint(rb.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                        //    Debug.DrawRay(body_T.position + rb.transform.TransformVector(rb.transform.localPosition), Vector3.up, Color.cyan,Time.fixedDeltaTime);
                        //    Debug.DrawRay(rb.transform.TransformPoint(rb.position), Vector3.up, Color.green,Time.fixedDeltaTime);


                        //                    Debug.DrawRay(body_T.position + body_T.InverseTransformVector(rb.transform.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                        //                    Debug.DrawRay(body_T.position + body_T.InverseTransformDirection(rb.transform.position), Vector3.up, Color.red,Time.fixedDeltaTime);
                        //    Debug.DrawRay(body_T.position + body_T.TransformPoint(rb.transform.localPosition), Vector3.up, Color.red,Time.fixedDeltaTime);

                        if (useTransformPoint)
                        {

                        }
                        else if (useTransformVector)
                        {

                        }
                        else if (useTransformDir)
                        {

                        }

                    }
                }

            }
        }


    }



    //    private void OnDrawGizmosSelected()
    private void OnDrawGizmos()
    {
        if (updateShiftCom)
        {
            foreach (var item in shiftComList)
            {
                item.rb.centerOfMass = item.shiftComAmount;
            }

            updateShiftCom = false;
        }
        if (!Application.isPlaying)
        {
            if (showCOMGizmos)
            {
                Vector3 CoM = Vector3.zero;
                float c = 0f;
                // avgCOMWorldSpace = Vector3.zero;
                //SHOW BODY PART GIZMOS
                foreach (var item in rbList)
                {
                    // if (item)
                    // {
                    Gizmos.color = bodyPartCOMColor;
                    float drawCOMRadius = item.mass / totalMass;
                    Gizmos.DrawSphere(item.worldCenterOfMass, drawCOMRadius);
                    //                        Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
                    CoM += item.worldCenterOfMass * item.mass;
                    c += item.mass;
                    // avgCOMWorldSpace += item.worldCenterOfMass;
                    // }
                }

                //DRAW AVG GIZMOS
                avgCOMWorldSpace = CoM / c;
                // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
                float avgCOMRadius = 0.5f; //radius of gizmo
                Gizmos.color = avgCOMColor;
                Gizmos.DrawWireSphere(avgCOMWorldSpace, avgCOMRadius);
            }
        }
        else
        {
            if (showCOMGizmos)
            {
                // avgCOMWorldSpace = Vector3.zero;

                //SHOW BODY PART GIZMOS
                foreach (var item in rbList)
                {
                    // if (item)
                    // {
                    Gizmos.color = bodyPartCOMColor;
                    float drawCOMRadius = item.mass / totalMass;
                    Gizmos.DrawSphere(item.worldCenterOfMass, drawCOMRadius);
                    //                        Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
                    // avgCOMWorldSpace += item.worldCenterOfMass;
                    // }
                }

                //DRAW AVG GIZMOS
                // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
                float avgCOMGizmoRadius = 0.5f; //radius of gizmo
                Gizmos.color = avgCOMColor;
                Gizmos.DrawWireSphere(avgCOMWorldSpace, avgCOMGizmoRadius);










            }

        }
    }
    // {
    //     if(!Application.isPlaying)
    //     {
    //         if (showCOMGizmos)
    //         {
    //             avgCOMWorldSpace = Vector3.zero;

    //             //SHOW BODY PART GIZMOS
    //             foreach(var item in rbList)
    //             {
    //                 if (item)
    //                 {
    //                     Gizmos.color = bodyPartCOMColor;
    //                     float drawCOMRadius = item.mass/totalMass;
    //                     Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
    //                     avgCOMWorldSpace += item.worldCenterOfMass;
    //                 }
    //             }

    //             //DRAW AVG GIZMOS
    //             avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
    //             float avgCOMRadius = 0.1f; //radius of gizmo
    //             Gizmos.color = avgCOMColor;
    //             Gizmos.DrawSphere(avgCOMWorldSpace, avgCOMRadius);
    //         }
    //     }
    //     else
    //     {
    //         if (showCOMGizmos)
    //         {
    //             // avgCOMWorldSpace = Vector3.zero;

    //             //SHOW BODY PART GIZMOS
    //             foreach(var item in rbList)
    //             {
    //                 if (item)
    //                 {
    //                     Gizmos.color = bodyPartCOMColor;
    //                     float drawCOMRadius = item.mass/totalMass;
    //                     Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
    //                     // avgCOMWorldSpace += item.worldCenterOfMass;
    //                 }
    //             }

    //             //DRAW AVG GIZMOS
    //             // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
    //             float avgCOMGizmoRadius = 0.1f; //radius of gizmo
    //             Gizmos.color = avgCOMColor;
    //             Gizmos.DrawSphere(avgCOMWorldSpace, avgCOMGizmoRadius);
    //         }

    //     }
    // }
}
