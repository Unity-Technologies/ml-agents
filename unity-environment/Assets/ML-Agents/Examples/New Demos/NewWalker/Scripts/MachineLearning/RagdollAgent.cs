using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RagdollAgent : Agent
{

    public Ragdoll ragdollPrefab;
    public Ragdoll ragdoll;

    public Light spotLight;

    Vector3 pelvisStartPos;
    Dictionary<Transform, Vector3> posDict;
    Dictionary<Transform, Quaternion> rotDict;

    public override void InitializeAgent()
    {
        base.InitializeAgent();

        if (ragdoll == null)
            ragdoll = GameObject.Instantiate(ragdollPrefab, transform.position, Quaternion.identity);
        posDict = new Dictionary<Transform, Vector3>();
        rotDict = new Dictionary<Transform, Quaternion>();
        foreach (var t in ragdoll.GetComponentsInChildren<Transform>())
        {
            posDict.Add(t, t.localPosition);
            rotDict.Add(t, t.localRotation);
        }
    }

    internal void AddVectorObs(Vector3 observation, List<float> state)
    {
        state.Add(observation.x);
        state.Add(observation.y);
        state.Add(observation.z);
    }

    void AddVectorObs(float obs, List<float> state)
    {
        state.Add(obs);
    }

    public Quaternion GetJointRotation(ConfigurableJoint joint)
    {
        return(Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
    }

    public void BodyPartObservation(LimbPiece bp)
    {
        var rb = bp.rigidbody;
        Vector3 localPosrelToHips = ragdoll.pelvis.transform.InverseTransformPoint(rb.position); //chilren of the hips are affected by it's scale this is a workaround to get the local pos rel to the hips

        // AddVectorObs(rb.transform.localPosition);
        AddVectorObs(localPosrelToHips);
        AddVectorObs(rb.position.y);
        AddVectorObs(rb.velocity);
        AddVectorObs(rb.angularVelocity);

        if(bp.joint)
        {
                var jointRotation = GetJointRotation(bp.joint);
            // if(bp.joint == joints[shinRJoint].joint)
            // {
            //     // float angle = Quaternion.Angle(joints[shinRJoint].joint.axis, joints[shinRJoint].joint.connectedBody.transform.rotation);
            //     float angle = Quaternion.Angle(shinRJoint.localRotation, joints[shinRJoint].joint.connectedBody.transform.localRotation);
            //     // float angle = Quaternion.Angle(joints[shinRJoint].rb.rotation, joints[shinRJoint].joint.connectedBody.rotation);
            //     print("Quat Angle" + angle);

            //     print("JointRotation: "  + jointRotation.eulerAngles);
            // }
                AddVectorObs(jointRotation); //get the joint rotation

        }
    }


    public override void CollectObservations()
    {
        AddVectorObs(ragdoll.pelvis.rigidbody.rotation);
        BodyPartObservation(ragdoll.pelvis);
        BodyPartObservation(ragdoll.head);
        BodyPartObservation(ragdoll.upperChest);
        BodyPartObservation(ragdoll.lowerChest);

        BodyPartObservation(ragdoll.leftUpperArm);
        BodyPartObservation(ragdoll.leftLowerArm);
        BodyPartObservation(ragdoll.leftHand);
        BodyPartObservation(ragdoll.rightUpperArm);
        BodyPartObservation(ragdoll.rightLowerArm);
        BodyPartObservation(ragdoll.rightHand);

        BodyPartObservation(ragdoll.leftUpperLeg);
        BodyPartObservation(ragdoll.leftLowerLeg);
        BodyPartObservation(ragdoll.leftFoot);
        BodyPartObservation(ragdoll.rightUpperLeg);
        BodyPartObservation(ragdoll.rightLowerLeg);
        BodyPartObservation(ragdoll.rightFoot);

        AddVectorObs(ragdoll.rightFoot.touchingGround ? 1f : 0f);
        AddVectorObs(ragdoll.leftFoot.touchingGround ? 1f : 0f);
        // AddVectorObs(ragdoll.rightHand.touchingGround ? 1f : 0f);
        // AddVectorObs(ragdoll.leftHand.touchingGround ? 1f : 0f);
    }

    // public override void CollectObservations()
    // {
    //     AddVectorObs(ragdoll.pelvis.Height);

    //     AddVectorObs(ragdoll.pelvis.transform.up);

    //     AddVectorObs(ragdoll.pelvis.transform.forward);

    //     AddVectorObs(ragdoll.pelvis.rigidbody.velocity);
    //     AddVectorObs(ragdoll.pelvis.rigidbody.angularVelocity);

    //     AddVectorObs(ragdoll.leftHand.LocalPosInPelvis);
    //     AddVectorObs(ragdoll.rightHand.LocalPosInPelvis);

    //     AddVectorObs(ragdoll.leftHand.Velocity);
    //     AddVectorObs(ragdoll.rightHand.Velocity);

    //     AddVectorObs(ragdoll.pelvis.transform.InverseTransformVector(ragdoll.leftHand.Velocity));
    //     AddVectorObs(ragdoll.pelvis.transform.InverseTransformVector(ragdoll.rightHand.Velocity));


    //     AddVectorObs(ragdoll.head.Height);
    //     AddVectorObs(ragdoll.head.transform.up);
    //     AddVectorObs(ragdoll.head.Velocity);
    //     AddVectorObs(ragdoll.head.rigidbody.angularVelocity);

    //     AddVectorObs(ragdoll.head.LocalPosInPelvis);
    //     // AddVectorObs(ragdoll.head.RelativePosFromPelvis);

    //     AddVectorObs(ragdoll.leftFoot.LocalPosInPelvis);
    //     AddVectorObs(ragdoll.rightFoot.LocalPosInPelvis);


    //     AddVectorObs(ragdoll.rightFoot.touchingGround ? 1f : 0f);
    //     AddVectorObs(ragdoll.leftFoot.touchingGround ? 1f : 0f);
    // }

    public bool useMuscleChain;

    public override void AgentAction(float[] act, string textAction)
    {

        if (useMuscleChain)
        {
            ragdoll.leftLegMuscleChain.SetTargetPos(new Vector3(act[0], act[1], act[2]));
            ragdoll.rightLefMuscleChain.SetTargetPos(new Vector3(act[3], act[4], act[5]));

            ragdoll.leftArmMuscle.SetTargetPos(new Vector3(act[6], act[7], act[8]));
            ragdoll.rightArmMuscle.SetTargetPos(new Vector3(act[9], act[10], act[11]));

            ragdoll.upperChest.SetNormalizedTargetRotation(act[12], act[13], act[14]);
            ragdoll.lowerChest.SetNormalizedTargetRotation(act[15], act[16], act[17]);

            ragdoll.leftFoot.SetNormalizedTargetRotation(act[18], act[19], act[20]);
            ragdoll.rightFoot.SetNormalizedTargetRotation(act[21], act[22], act[23]);
        }
        else
        {
            ragdoll.leftUpperLeg.SetNormalizedTargetRotation(act[0], act[1], act[2]);
            ragdoll.leftLowerLeg.SetNormalizedTargetRotation(act[3], act[4], act[5]);

            ragdoll.rightUpperLeg.SetNormalizedTargetRotation(act[6], act[7], act[8]);
            ragdoll.rightLowerLeg.SetNormalizedTargetRotation(act[9], act[10], act[11]);

            ragdoll.leftUpperArm.SetNormalizedTargetRotation(act[12], act[13], act[14]);
            ragdoll.leftLowerArm.SetNormalizedTargetRotation(act[15], act[16], act[17]);

            ragdoll.rightUpperArm.SetNormalizedTargetRotation(act[18], act[19], act[20]);
            ragdoll.rightLowerArm.SetNormalizedTargetRotation(act[21], act[22], act[23]);

            ragdoll.upperChest.SetNormalizedTargetRotation(act[24], act[25], act[26]);
            ragdoll.lowerChest.SetNormalizedTargetRotation(act[27], act[28], act[29]);

            ragdoll.leftFoot.SetNormalizedTargetRotation(act[30], act[31], act[32]);
            ragdoll.rightFoot.SetNormalizedTargetRotation(act[33], act[34], act[35]);
        }

        //if (float.IsNaN(ragdoll.head.Height) || float.IsInfinity(ragdoll.head.Height) || ragdoll.head.Height > 5f || ragdoll.head.Height < 1f)
        //{
        //    reward = -1f;
        //    done = true;
        //}
        //else
        {




        AddReward(ragdoll.head.Height * .01f);
        AddReward(ragdoll.head.rigidbody.velocity.sqrMagnitude * -.001f);
            // SetReward((ragdoll.head.Height - 1.2f) + ragdoll.head.transform.up.y * 0.1f);

            // if (ragdoll.upperChest.touchingGround || ragdoll.lowerChest.touchingGround || ragdoll.head.touchingGround || ragdoll.head.Height < 1.2f)
            // {
            //     SetReward(-1f);
            //     if (Application.isEditor)
            //         print(GetCumulativeReward());
            //     Done();
            // }



        }




    }
    public override void AgentReset()
    {

        if (ragdoll != null)
        {
            foreach (var t in ragdoll.GetComponentsInChildren<Transform>())
            {
                t.localPosition = posDict[t];
                t.localRotation = rotDict[t];
                if (t.GetComponent<Rigidbody>() != null)
                {
                    t.GetComponent<Rigidbody>().velocity = t.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
                }
            }
        }

        if (spotLight != null)
        {
            spotLight.transform.LookAt(transform.position + Random.insideUnitSphere * 2f, transform.forward);
            spotLight.intensity = Random.Range(5f, 45f);
            spotLight.color = new Color(Random.value, Random.value, Random.value);
            spotLight.spotAngle = Random.Range(20f, 60f);
        }

    }

    public override void AgentOnDone()
    {

    }


}
