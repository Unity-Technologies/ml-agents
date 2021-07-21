//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class LaserAgentDiverse : Agent
{

    public GameObject goal;
    Rigidbody m_AgentRb;

    public float m_AgentSpeed = 0.5f;

    private bool initCrouch = false;
    private bool finishCrouch = false;
    private bool initJump = false;
    private bool finishJump = false;
    private float lastDist = 0;
    private float initDist = 0;

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        float startZ = Random.Range(-1f, 1f);
        float startRot = Random.Range(20f, 160f);
        transform.localPosition = new Vector3(-4.65f, 0.225f, startZ);
        transform.localEulerAngles = new Vector3(0f, startRot, 0f);
        transform.localScale = new Vector3(0.25f, 0.25f, 0.25f);
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;
        initJump = false;
        finishJump = false;
        initCrouch = false;
        finishCrouch = false;

        initDist = GetDistToGoal();
        lastDist = initDist;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float centeredAngle = transform.localEulerAngles.y < 270 ? 
                              transform.localEulerAngles.y : 
                              transform.localEulerAngles.y - 360f;
        sensor.AddObservation(transform.localPosition.y);
        sensor.AddObservation(centeredAngle / 180f - 0.5f);
        sensor.AddObservation(transform.localScale.y * 4);
        sensor.AddObservation(initCrouch ? 1 : finishCrouch ? -1 : 0);
        sensor.AddObservation(initJump ? 1 : finishJump ? -1 : 0);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        int moveAction = actionBuffers.DiscreteActions[0];
        int specialAction = actionBuffers.DiscreteActions[1];
        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        switch (moveAction)
        {
            case 1:
                dirToGo += transform.forward;
                break;
            case 2:
                dirToGo -= transform.forward;
                break;
            case 3:
                dirToGo += transform.right;
                break;
            case 4:
                dirToGo -= transform.right;
                break;
            case 5:
                rotateDir += transform.up;
                break;
            case 6:
                rotateDir -= transform.up;
                break;
        }
        if (!initJump && !finishJump && !initCrouch && !finishCrouch) 
        {
            switch (specialAction)
            {
                case 1:
                    initJump = true;
                    break;
                case 2:
                    initCrouch = true;
                    break;
            }
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_AgentSpeed, ForceMode.VelocityChange);

        float dist = GetDistToGoal();
        AddReward((lastDist - dist) / initDist);
        AddReward(-1f / MaxStep);
        lastDist = dist;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        if (Input.GetKey(KeyCode.E))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            discreteActionsOut[0] = 4;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 5;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 6;
        }
        if (Input.GetKey(KeyCode.Space))
        {
            discreteActionsOut[1] = 1;
        }
        else if (Input.GetKey(KeyCode.LeftShift))
        {
            discreteActionsOut[1] = 2;
        }
    }

    void FixedUpdate()
    {
        float crouch = transform.localScale.y;
        if (initCrouch && crouch > 0.1f) 
        {
            transform.localScale -= new Vector3(0f, 0.01f, 0f);
            m_AgentRb.AddForce(transform.forward * m_AgentSpeed, ForceMode.VelocityChange);
        }
        else if (initCrouch)
        {
            initCrouch = false;
            finishCrouch = true;
        }
        else if (finishCrouch && crouch < 0.25)
        {
            transform.localScale += new Vector3(0f, 0.01f, 0f);
        }
        else
        {
            initCrouch = false;
            finishCrouch = false;
            transform.localScale = new Vector3(0.25f, 0.25f, 0.25f);
        }

        float jump = transform.localPosition.y;
        if (initJump && jump < .5f)
        {
            m_AgentRb.AddForce(transform.up + transform.forward * m_AgentSpeed, ForceMode.VelocityChange);
        }
        else if (initJump)
        {
            initJump = false;
            finishJump = true;
        }
        else if (finishJump && jump <= 0.23f)
        {
            initJump = false;
            finishJump = false;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == goal)
        {
            // AddReward(1f);
            EndEpisode();
        }
        else if (other.tag == "deadly") 
        {
            // AddReward(-1f);
            EndEpisode();
        }
    }

    private float GetDistToGoal()
    {
        return Vector3.Project(goal.transform.position - transform.position, goal.transform.right).magnitude; 
    }
}
