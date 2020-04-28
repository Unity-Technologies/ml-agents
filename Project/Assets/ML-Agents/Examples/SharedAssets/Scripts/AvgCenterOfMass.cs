using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Used for visualizing the average center of mass of a ragdoll
/// </summary>
[DisallowMultipleComponent]
[ExecuteInEditMode]
public class AvgCenterOfMass : MonoBehaviour
{
    /// <summary>
    /// Enable to show a green spehere at the current center of mass.
    /// </summary>
    [Tooltip("Enable to show a green spehere at the current center of mass.")]
    public bool showCOMGizmos = true;
    public Vector3 avgCOMWorldSpace;
    public Color avgCOMColor = Color.green;
    public Color bodyPartCOMColor = Color.yellow;
    List<Rigidbody> rbList = new List<Rigidbody>();
    public float totalMass;

    void Start()
    {
        SetUpRigidbodies();
    }

    void SetUpRigidbodies()
    {
        rbList.Clear();
        totalMass = 0;
        foreach(var item in GetComponentsInChildren<Rigidbody>())
        {
            rbList.Add(item);
            totalMass += item.mass;
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
    void FixedUpdate()
    {
        if(Application.isPlaying)
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


            Vector3 CoM = Vector3.zero;
            float c = 0f;
            
            foreach(var item in rbList)
            {
                CoM += item.worldCenterOfMass * item.mass;
                c += item.mass;
            }
            avgCOMWorldSpace = CoM/c;
            // CoM /= c;
        }
    }



    private void OnDrawGizmosSelected()
    {
        if(!Application.isPlaying)
        {
            if (showCOMGizmos)
            {
                Vector3 CoM = Vector3.zero;
                float c = 0f;
                // avgCOMWorldSpace = Vector3.zero;
                //SHOW BODY PART GIZMOS
                foreach(var item in rbList)
                {
                    // if (item)
                    // {
                        Gizmos.color = bodyPartCOMColor;
                        float drawCOMRadius = item.mass/totalMass;
                        Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
                        CoM += item.worldCenterOfMass * item.mass;
                        c += item.mass;
                        // avgCOMWorldSpace += item.worldCenterOfMass;
                    // }
                }

                //DRAW AVG GIZMOS
                avgCOMWorldSpace = CoM/c;
                // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
                float avgCOMRadius = 0.1f; //radius of gizmo 
                Gizmos.color = avgCOMColor;
                Gizmos.DrawSphere(avgCOMWorldSpace, avgCOMRadius);
            }
        }
        else
        {
            if (showCOMGizmos)
            {
                // avgCOMWorldSpace = Vector3.zero;

                //SHOW BODY PART GIZMOS
                foreach(var item in rbList)
                {
                    // if (item)
                    // {
                        Gizmos.color = bodyPartCOMColor;
                        float drawCOMRadius = item.mass/totalMass;
                        Gizmos.DrawWireSphere(item.worldCenterOfMass, drawCOMRadius);
                        // avgCOMWorldSpace += item.worldCenterOfMass;
                    // }
                }

                //DRAW AVG GIZMOS
                // avgCOMWorldSpace /= rbList.Count; //divide by num of rb's to get avg in WORLD space
                float avgCOMGizmoRadius = 0.1f; //radius of gizmo 
                Gizmos.color = avgCOMColor;
                Gizmos.DrawSphere(avgCOMWorldSpace, avgCOMGizmoRadius);
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
