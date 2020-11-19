using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class FPSAgent : Agent
{
    private AgentCubeMovement m_CubeMovement;

    public ShootProjectiles shootController;
    public bool useVectorObs;
    Rigidbody m_AgentRb;
    bool m_Shoot;
    private Camera m_Cam;

    // Start is called before the first frame update
    public override void Initialize()
    {
        m_CubeMovement = GetComponent<AgentCubeMovement>();
        m_Cam = Camera.main;
        m_AgentRb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        //        Unfreeze();
        //        Unpoison();
        //        Unsatiate();
        //        m_Shoot = false;
        //        m_AgentRb.velocity = Vector3.zero;
        //        myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        //        transform.position = new Vector3(Random.Range(-m_MyArea.range, m_MyArea.range),
        //                                 2f, Random.Range(-m_MyArea.range, m_MyArea.range))
        //                             + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        //        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVectorObs)
        {
            //            var localVelocity = transform.InverseTransformDirection(m_AgentRb.velocity);
            //            sensor.AddObservation(localVelocity.x);
            //            sensor.AddObservation(localVelocity.z);
            //            sensor.AddObservation(m_Frozen);
            sensor.AddObservation(m_Shoot);
        }
        //        else if (useVectorFrozenFlag)
        //        {
        //            sensor.AddObservation(m_Frozen);
        //        }
    }

    public void MoveAgent(ActionSegment<float> act)
    {

        //        if (!m_Frozen)
        //        {
        //            var shootCommand = false;
        var forwardAxis = act[0];
        var rightAxis = act[1];
        var rotateAxis = act[2];
        var shootAxis = act[3];
        m_Shoot = shootAxis > 0 ? true : false;

        m_CubeMovement.RunOnGround(m_AgentRb, m_Cam.transform.TransformDirection(new Vector3(rightAxis, 0, forwardAxis)));
        m_CubeMovement.RotateBody(rotateAxis, forwardAxis);
        if (shootAxis > 0)
        {
            shootController.Shoot();
        }
        //        }

        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("projectile"))
        {
            //IMPLEMENT HEALTH MECHANIC
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.ContinuousActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = Input.GetKey(KeyCode.W) ? 1 : Input.GetKey(KeyCode.S) ? -1 : 0; //inputV
        contActionsOut[1] = Input.GetKey(KeyCode.E) ? 1 : Input.GetKey(KeyCode.Q) ? -1 : 0; //inputH
        contActionsOut[2] = Input.GetKey(KeyCode.D) ? 1 : Input.GetKey(KeyCode.A) ? -1 : 0; //rotate
        contActionsOut[3] = Input.GetKey(KeyCode.Space) ? 1 : 0; //shoot

    }

}
