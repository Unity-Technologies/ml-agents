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

    public MultiGunAlternating gunController;
    public bool useVectorObs;
    Rigidbody m_AgentRb;
    //    bool m_Shoot;
    private Camera m_Cam;
    [Header("HEALTH")] public AgentHealth agentHealth;

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
            sensor.AddObservation(m_ShootInput);
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
        //        var forwardAxis = act[0];
        //        var rightAxis = act[1];
        //        var rotateAxis = act[2];
        //        var shootAxis = act[3];
        //        m_Shoot = shootAxis > 0;

        //        m_CubeMovement.RotateBody(rotateAxis, forwardAxis);
        //        m_CubeMovement.RunOnGround(m_AgentRb, m_Cam.transform.TransformDirection(new Vector3(0, 0, forwardAxis)));
        //        m_CubeMovement.Strafe(transform.right * rightAxis);

        m_InputV = act[0];
        m_InputH = act[1];
        m_Rotate = act[2];
        m_ShootInput = act[3];
        m_CubeMovement.RotateBody(m_Rotate, m_InputV);
        m_CubeMovement.RunOnGround(m_AgentRb, m_Cam.transform.TransformDirection(new Vector3(0, 0, m_InputV)));
        //        if (m_InputH != 0)
        //        {

        //        if (leftStrafe)
        //        {
        //            m_CubeMovement.Strafe(transform.right * -1);
        //            leftStrafe = false;
        //        }
        //        if (rightStrafe)
        //        {
        //            m_CubeMovement.Strafe(transform.right * 1);
        //            rightStrafe = false;
        //        }
        //
        m_CubeMovement.Strafe(transform.right * m_InputH);
        //        }
        if (m_ShootInput > 0)
        {
            gunController.Shoot();
        }
        //        }

        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

    }

    //    void OnCollisionEnter(Collision col)
    //    {
    //        if (col.gameObject.CompareTag("projectile"))
    //        {
    //            //IMPLEMENT HEALTH MECHANIC
    //        }
    //    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.ContinuousActions);
    }

    public float m_InputH;
    private float m_InputV;
    private float m_Rotate;
    private float m_ShootInput;
    public bool leftStrafe;
    public bool rightStrafe;
    void Update()
    {
        //        m_InputH = Input.GetKeyDown(KeyCode.K) ? 1 : Input.GetKeyDown(KeyCode.J) ? -1 : 0; //inputH
        if (Input.GetKeyDown(KeyCode.K))
        {
            rightStrafe = true;
        }
        if (Input.GetKeyDown(KeyCode.J))
        {
            leftStrafe = true;
        }
    }

    void FixedUpdate()
    {
        m_InputV = Input.GetKey(KeyCode.W) ? 1 : Input.GetKey(KeyCode.S) ? -1 : 0; //inputV
                                                                                   //        m_InputH = 0;
                                                                                   //        m_InputH += Input.GetKeyDown(KeyCode.Q) ? -1 : 0;
                                                                                   //        m_InputH += Input.GetKeyDown(KeyCode.E) ? 1 : 0;
                                                                                   //        m_InputH = Input.GetKeyDown(KeyCode.E) ? 1 : Input.GetKeyDown(KeyCode.Q) ? -1 : 0; //inputH
        m_Rotate = 0;
        m_Rotate += Input.GetKey(KeyCode.A) ? -1 : 0;
        m_Rotate += Input.GetKey(KeyCode.D) ? 1 : 0;
        //        m_Rotate = Input.GetKey(KeyCode.D) ? 1 : Input.GetKey(KeyCode.A) ? -1 : 0; //rotate
        m_ShootInput = Input.GetKey(KeyCode.Space) ? 1 : 0; //shoot
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var contActionsOut = actionsOut.ContinuousActions;
        contActionsOut[0] = m_InputV; //inputV
        contActionsOut[2] = m_Rotate; //rotate
        contActionsOut[3] = m_ShootInput; //shoot
                                          //        contActionsOut[0] = Input.GetKey(KeyCode.W) ? 1 : Input.GetKey(KeyCode.S) ? -1 : 0; //inputV
                                          //        contActionsOut[1] = Input.GetKeyDown(KeyCode.E) ? 1 : Input.GetKeyDown(KeyCode.Q) ? -1 : 0; //inputH
                                          //        contActionsOut[2] = Input.GetKey(KeyCode.D) ? 1 : Input.GetKey(KeyCode.A) ? -1 : 0; //rotate
                                          //        contActionsOut[3] = Input.GetKey(KeyCode.Space) ? 1 : 0; //shoot

        m_InputH = 0;
        if (leftStrafe)
        {
            //            print("leftstrafe");
            m_InputH += -1;
            leftStrafe = false;
        }
        if (rightStrafe)
        {
            //            print("rightstrafe");
            m_InputH += 1;
            rightStrafe = false;
        }
        contActionsOut[1] = m_InputH; //inputH
    }

}
