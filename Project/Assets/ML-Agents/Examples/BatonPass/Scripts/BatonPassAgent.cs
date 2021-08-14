using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class BatonPassAgent : Agent
{
    BatonPassArea m_Area;
    Rigidbody m_AgentRb;

    public bool Dead;

    public int Lifetime = 400;
    int m_Lifetime;

    public override void Initialize()
    {
        m_Lifetime = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("agent_steps", Lifetime);
        m_AgentRb = GetComponent<Rigidbody>();
        Dead = false;
    }

    public void SetLife(int remaining)
    {
        m_Lifetime = remaining;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!Dead)
        {
            // sensor.AddObservation(1.0f * m_Area.GetTimeLeft());
            sensor.AddObservation(transform.InverseTransformDirection(m_AgentRb.velocity));

            sensor.AddObservation(2.0f * m_Lifetime / Academy.Instance.EnvironmentParameters.GetWithDefault("agent_steps", Lifetime) - 1f);
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                sensor.AddObservation(0f);
            }
        }
    }

    public void SetArea(BatonPassArea area)
    {
        m_Area = area;
    }

    void FixedUpdate()
    {
        m_Lifetime -= 1;
        if (m_Lifetime < 0)
        {
            if ((int)Academy.Instance.EnvironmentParameters.GetWithDefault("absorbing_state", 0f) == 0f)
            {
                m_Area.UnregisterAgent(this.gameObject, true);
                Destroy(this.gameObject);
            }
            else
            {
                Vector3 pos = new Vector3(Random.Range(200f, 2000f), Random.Range(-2000f, -1000f), Random.Range(-1000f, 1000f));
                var rot = Quaternion.Euler(Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f));
                transform.SetPositionAndRotation(pos, rot);
                Dead = true;
                m_Lifetime = System.Int32.MaxValue;
                m_Area.UnregisterAgent(this.gameObject, false);
            }
        }
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var actionForward = act[0];
        var actionRotate = act[1];
        switch (actionForward)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
        }
        switch (actionRotate)
        {
            case 1:
                rotateDir = transform.up * 1f;
                break;
            case 2:
                rotateDir = transform.up * -1f;
                break;
        }
        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        m_AgentRb.velocity = dirToGo * 10;
        // m_AgentRb.AddForce(dirToGo * 2f, ForceMode.VelocityChange);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 2;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }
}
