//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class PushAgentZombie : Agent
{

    public GameObject MyKey;
    public bool IHaveAKey;
    private PushBlockSettings m_PushBlockSettings;
    private Rigidbody m_AgentRb;  //cached on initialization
    private ZombiePushBlockDeathEnvController m_GameController;

    public override void Initialize()
    {
        m_GameController = GetComponentInParent<ZombiePushBlockDeathEnvController>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        MyKey.SetActive(false);
        IHaveAKey = false;
    }

    public override void OnEpisodeBegin()
    {
        MyKey.SetActive(false);
        IHaveAKey = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // if (useVectorObs)
        // {
        sensor.AddObservation(IHaveAKey);
        // sensor.AddObservation(m_GameController.PlayerDict[this].HoldingSwitch);
        // sensor.AddObservation(m_GameController.PlayerDict[this].Scored);
        // }
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Move the agent using the action.
        MoveAgent(actionBuffers.DiscreteActions);
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag("lock"))
        {
            if (IHaveAKey)
            {
                m_GameController.UnlockBlock(col.transform);
                MyKey.SetActive(false);
                IHaveAKey = false;
            }
        }
        if (col.transform.CompareTag("zombie"))
        {
            m_GameController.KilledByZombie(this, col);
            MyKey.SetActive(false);
            IHaveAKey = false;
        }
    }

    void OnTriggerEnter(Collider col)
    {
        //if we find a key and it's parent is the main platform we can pick it up
        // if (col.transform.CompareTag("key"))
        if (col.transform.CompareTag("key") && col.transform.parent == transform.parent && gameObject.activeInHierarchy)
        {
            print("picked up key");
            MyKey.SetActive(true);
            IHaveAKey = true;
            col.gameObject.SetActive(false);
            // DestroyImmediate(col.gameObject);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }
}
