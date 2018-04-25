using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WalkerAgentMotorJoints : Agent
{

    public float strength;
    float x_position;
    [HideInInspector]
    public bool[] leg_touching;
    [HideInInspector]
    public bool fell;
    Vector3 past_velocity;
    Transform body;
    Rigidbody bodyRB;
    public Transform[] limbs;
    // public ConfigurableJoint[] joints;
    public Rigidbody[] limbRBs;
    Dictionary<GameObject, Vector3> transformsPosition;
    Dictionary<GameObject, Quaternion> transformsRotation;
    public float totalCharMass; //total mass of this agent
    public bool visualizeMassDistribution;

    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;
    public Transform chestJoint;
    public Transform spineJoint;
    public Transform thighLJoint;
    public Transform shinLJoint;
    public Transform thighRJoint;
    public Transform shinRJoint;
    public Transform armLJoint;
    public Transform forearmLJoint;
    public Transform armRJoint;
    public Transform forearmRJoint;



    public Dictionary<Transform, BodyPart> bodyParts = new Dictionary<Transform, BodyPart>();
    public Dictionary<Transform, BodyPart> joints = new Dictionary<Transform, BodyPart>();
    public float agentEnergy = 100;
    public float energyRegenerationRate;

    [System.Serializable]
    public class BodyPart
    {
        public ConfigurableJoint joint;
        public Rigidbody rb;
        public Vector3 startingPos;
        public Quaternion startingRot;
        public float currentEnergyLevel;
    }

    public void SetupBodyPart(Transform t)
    {
        BodyPart bp = new BodyPart();
        bp.rb = t.GetComponent<Rigidbody>();
        bp.joint = t.GetComponent<ConfigurableJoint>();
        bp.startingPos = t.position;
        bp.startingRot = t.rotation;
        bodyParts.Add(t, bp);
    }
    public void SetupMotorJoint(Transform t)
    {
        BodyPart bp = new BodyPart();
        bp.rb = t.GetComponent<Rigidbody>();
        bp.joint = t.GetComponent<ConfigurableJoint>();
        bp.startingPos = t.position;
        bp.startingRot = t.rotation;
        joints.Add(t, bp);
    }

    public override void InitializeAgent()
    {
        SetupBodyPart(hips);
        SetupBodyPart(chest);
        SetupBodyPart(spine);
        SetupBodyPart(head);
        SetupBodyPart(thighL);
        SetupBodyPart(shinL);
        SetupBodyPart(footL);
        SetupBodyPart(thighR);
        SetupBodyPart(shinR);
        SetupBodyPart(footR);
        SetupBodyPart(armL);
        SetupBodyPart(forearmL);
        SetupBodyPart(handL);
        SetupBodyPart(armR);
        SetupBodyPart(forearmR);
        SetupBodyPart(handR);

        SetupMotorJoint(chestJoint);
        SetupMotorJoint(spineJoint);
        SetupMotorJoint(thighLJoint);
        SetupMotorJoint(shinLJoint);
        SetupMotorJoint(thighRJoint);
        SetupMotorJoint(shinRJoint);
        SetupMotorJoint(armLJoint);
        SetupMotorJoint(forearmLJoint);
        SetupMotorJoint(armRJoint);
        SetupMotorJoint(forearmRJoint);

        // body = transform.Find("Body");
        // bodyRB = body.GetComponent<Rigidbody>();
        transformsPosition = new Dictionary<GameObject, Vector3>();
        transformsRotation = new Dictionary<GameObject, Quaternion>();
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            transformsPosition[child.gameObject] = child.position;
            transformsRotation[child.gameObject] = child.rotation;
        }
        leg_touching = new bool[2];
        limbRBs = new Rigidbody[limbs.Length];
        totalCharMass = 0; //reset to 0
        // for (int i = 0; i < limbs.Length; i++)
        // {
        //     limbRBs[i] = limbs[i].gameObject.GetComponent<Rigidbody>();
        //     joints[i] = limbs[i].gameObject.GetComponent<ConfigurableJoint>();
        //     // if(limbRBs[i])
        //     // {
        //     //     limbRBs[i].maxAngularVelocity = 50;
        //     //     limbRBs[i].centerOfMass += limbRBs[i].transform.TransformPoint(joints[i].anchor);
        //     //     totalCharMass += limbRBs[i].mass;
        //     // }
        // }

        foreach(var item in bodyParts)
        {
            if(item.Value.rb)
            {
                  item.Value.rb.maxAngularVelocity = 500;
                // if(joints[i])
                // limbRBs[i].centerOfMass += joints[i].anchor;
                // if(item.Value.joint)
                // {
                //     item.Value.rb.centerOfMass += Vector3.Scale(item.Value.joint.anchor, item.Value.rb.transform.localScale);
                // }
                totalCharMass += item.Value.rb.mass;
            }
        }
        foreach(var item in joints)
        {
            if(item.Value.rb)
            {
                  item.Value.rb.maxAngularVelocity = 500;
                // if(joints[i])
                // limbRBs[i].centerOfMass += joints[i].anchor;
                // if(item.Value.joint)
                // {
                //     item.Value.rb.centerOfMass += Vector3.Scale(item.Value.joint.anchor, item.Value.rb.transform.localScale);
                // }
                totalCharMass += item.Value.rb.mass;
            }
        }
        // for (int i = 0; i < limbs.Length; i++)
        // {
        //     // limbRBs[i] = limbs[i].gameObject.GetComponent<Rigidbody>();
        //     // joints[i] = limbs[i].gameObject.GetComponent<ConfigurableJoint>();
        //     if(limbRBs[i])
        //     {
        //         limbRBs[i].maxAngularVelocity = 50;
        //         // if(joints[i])
        //         // limbRBs[i].centerOfMass += joints[i].anchor;
        //         limbRBs[i].centerOfMass += Vector3.Scale(joints[i].anchor, limbRBs[i].transform.localScale);
        //         totalCharMass += limbRBs[i].mass;
        //     }
        // }
    }
    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return(Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    public void BodyPartObservation(BodyPart bp)
    {
        var rb = bp.rb;
        Vector3 localPosrelToHips = hips.InverseTransformPoint(bp.rb.position); //chilren of the hips are affected by it's scale this is a workaround to get the local pos rel to the hips

        // AddVectorObs(rb.transform.localPosition);
        AddVectorObs(localPosrelToHips);
        AddVectorObs(rb.position.y);
        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);

        if(bp.joint)
        {
                var jointRotation = GetJointRotation(bp.joint);
            if(bp.joint == joints[shinRJoint].joint)
            {
                // float angle = Quaternion.Angle(joints[shinRJoint].joint.axis, joints[shinRJoint].joint.connectedBody.transform.rotation);
                float angle = Quaternion.Angle(shinRJoint.localRotation, joints[shinRJoint].joint.connectedBody.transform.localRotation);
                // float angle = Quaternion.Angle(joints[shinRJoint].rb.rotation, joints[shinRJoint].joint.connectedBody.rotation);
                print("Quat Angle" + angle);

                print("JointRotation: "  + jointRotation.eulerAngles);
            }
                AddVectorObs(jointRotation); //get the joint rotation

        }
    }

    public override void CollectObservations()
    {
        AddVectorObs(bodyParts[hips].rb.rotation);

        BodyPartObservation(joints[chestJoint]);
        BodyPartObservation(joints[spineJoint]);

        BodyPartObservation(joints[thighLJoint]);
        BodyPartObservation(joints[shinLJoint]);
        BodyPartObservation(joints[thighRJoint]);
        BodyPartObservation(joints[shinRJoint]);

        BodyPartObservation(joints[armLJoint]);
        BodyPartObservation(joints[forearmLJoint]);
        BodyPartObservation(joints[armRJoint]);
        BodyPartObservation(joints[forearmRJoint]);
        BodyPartObservation(bodyParts[hips]);
        BodyPartObservation(bodyParts[handL]);
        BodyPartObservation(bodyParts[handR]);
        BodyPartObservation(bodyParts[footR]);
        BodyPartObservation(bodyParts[footL]);
        BodyPartObservation(bodyParts[head]);


        for (int index = 0; index < 2; index++)
        {
            if (leg_touching[index])
            {
                AddVectorObs(1);
            }
            else
            {
                AddVectorObs(0);
            }
            leg_touching[index] = false;
        }
    }
    
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        float[] toUse = new float[vectorAction.Length];
        for (int k = 0; k < vectorAction.Length; k++)
        {
            toUse[k] = Mathf.Clamp(vectorAction[k], -3f, 3f);
        }
        
        
        ForceMode forceModeToUse = ForceMode.Force;
        joints[shinLJoint].rb.AddTorque(shinLJoint.right * strength * toUse[0], forceModeToUse);

        // joints[thighLJoint].rb.AddTorque(thighLJoint.right * strength * toUse[0], forceModeToUse);
        // joints[thighRJoint].rb.AddTorque(thighRJoint.right * strength * toUse[1], forceModeToUse);
        // joints[thighLJoint].rb.AddTorque(thighLJoint.forward * strength * toUse[2], forceModeToUse);
        // joints[thighRJoint].rb.AddTorque(thighRJoint.forward * strength * toUse[3], forceModeToUse);
        // joints[shinLJoint].rb.AddTorque(shinLJoint.right * strength * toUse[4], forceModeToUse);
        // joints[shinRJoint].rb.AddTorque(shinRJoint.right * strength * toUse[5], forceModeToUse);
        // joints[spineJoint].rb.AddTorque(chestJoint.right * strength * toUse[6], forceModeToUse);
        // joints[spineJoint].rb.AddTorque(chestJoint.forward * strength * toUse[7], forceModeToUse);
        // joints[chestJoint].rb.AddTorque(chestJoint.right * strength * toUse[8], forceModeToUse);
        // joints[chestJoint].rb.AddTorque(chestJoint.forward * strength * toUse[9], forceModeToUse);
        // joints[armLJoint].rb.AddTorque(armLJoint.forward * strength * toUse[10], forceModeToUse);
        // joints[armLJoint].rb.AddTorque(armLJoint.right * strength * toUse[11], forceModeToUse);
        // joints[armRJoint].rb.AddTorque(armRJoint.forward * strength * toUse[12], forceModeToUse);
        // joints[armRJoint].rb.AddTorque(armRJoint.right * strength * toUse[13], forceModeToUse);
        // joints[forearmRJoint].rb.AddTorque(forearmRJoint.right * strength * toUse[14], forceModeToUse);
        // joints[forearmLJoint].rb.AddTorque(forearmLJoint.right * strength * toUse[15], forceModeToUse);

        float torquePenalty = 0; 
        // for (int k = 0; k < 15; k++)
        // {
        //     torquePenalty += toUse[k] * toUse[k];
        // }
        
        AddReward(
            - 0.001f * torquePenalty
            + 0.02f * Mathf.Clamp(bodyParts[hips].rb.velocity.x, 0f, 1000f)
            + 0.01f * bodyParts[chest].rb.position.y
        );
    }

    void OnDrawGizmos()
    {
        if(Application.isPlaying)
        {
            if(visualizeMassDistribution)
            {
                // Gizmos.color = new Color(0,1,1,.5f);
                float drawCOMRadius = 0; //our center of mass radius is relative to the mass of the body part's proportional mass vs the whole body
                totalCharMass = 0;
                foreach(var item in bodyParts)
                {
                    if(item.Value.rb)
                    {
                        totalCharMass += item.Value.rb.mass;
                    }
                }
                foreach(var item in bodyParts)
                {
                    if(item.Value.rb)
                    {
                        Gizmos.color = new Color(0,1,1,.5f);
                        drawCOMRadius = item.Value.rb.mass/totalCharMass;
                        // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(limbRBs[i].transform.up + joints[i].anchor);
                        // var COMPosition = limbRBs[i].transform.TransformPoint(joints[i].anchor);
                        var COMPosition = item.Value.rb.worldCenterOfMass;
                        // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(joints[i].anchor);
                        Gizmos.DrawSphere(COMPosition, drawCOMRadius);

                        Gizmos.color = Color.yellow;
                        Vector3 localPosrelToHips = hips.InverseTransformPoint(item.Key.position); //chilren of the hips are affected by it's scale this is a workaround to get the local pos rel to the hips
                        Gizmos.DrawSphere(localPosrelToHips, drawCOMRadius);
                        Gizmos.color = Color.green;
                        Vector3 worldPosrelToHipsPoint = hips.TransformPoint(localPosrelToHips);
                        Gizmos.DrawSphere(worldPosrelToHipsPoint, drawCOMRadius);
                        // Gizmos.DrawSphere(item.Value.rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);

                        // // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Value.rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Key.transform.TransformPoint(bodyParts[hips].rb.worldCenterOfMass), drawCOMRadius);
                        // Gizmos.DrawSphere(item.Value.rb.position, drawCOMRadius);
                        // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass, drawCOMRadius);


                    }
                }
                // foreach(var item in joints)
                // {
                //     if(item.Value.rb)
                //     {
                //         // Gizmos.color = new Color(0,1,1,.5f);
                //         Gizmos.color = Color.red;
                //         drawCOMRadius = item.Value.rb.mass/totalCharMass;
                //         // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(limbRBs[i].transform.up + joints[i].anchor);
                //         // var COMPosition = limbRBs[i].transform.TransformPoint(joints[i].anchor);
                //         var COMPosition = item.Value.rb.worldCenterOfMass;
                //         // var COMPosition = limbRBs[i].worldCenterOfMass + limbRBs[i].transform.TransformPoint(joints[i].anchor);
                //         Gizmos.DrawSphere(COMPosition, drawCOMRadius);

                //         // // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                //         // Gizmos.DrawSphere(item.Value.rb.worldCenterOfMass + (bodyParts[hips].rb.worldCenterOfMass - item.Value.rb.worldCenterOfMass), drawCOMRadius);
                //         // Gizmos.DrawSphere(item.Key.transform.TransformPoint(bodyParts[hips].rb.worldCenterOfMass), drawCOMRadius);
                //         // Gizmos.DrawSphere(item.Value.rb.position, drawCOMRadius);
                //         // Gizmos.DrawSphere(bodyParts[hips].rb.worldCenterOfMass, drawCOMRadius);


                //     }
                // }
                
            }

        }
    }

    public override void AgentReset()
    {
        print("reset");
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            child.position = transformsPosition[child.gameObject];
            child.rotation = transformsRotation[child.gameObject];
            if (child.gameObject.GetComponent<Rigidbody>())
            {
                child.gameObject.GetComponent<Rigidbody>().velocity = default(Vector3);
                child.gameObject.GetComponent<Rigidbody>().angularVelocity = default(Vector3);
            }
        }
        // gameObject.transform.rotation = Quaternion.Euler(new Vector3(0, 90f, 0));
    }

    public override void AgentOnDone()
    {

    }
}
