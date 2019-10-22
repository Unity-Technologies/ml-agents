using System;
using MLAgents;
using UnityEngine;

public class ManualControlReacherArm : MonoBehaviour
{
    public GameObject pendulumA;
    public GameObject pendulumB;
    public GameObject hand;
    public GameObject goal;
    private ArticulationBody m_RbA;
    private ArticulationBody m_RbB;

    private Vector3 m_TorqueA;
    private Vector3 m_TorqueB;

    /// <summary>
    /// Collect the rigidbodies of the reacher in order to resue them for
    /// observations and actions.
    /// </summary>
    public void Start()
    {
        m_RbA = pendulumA.GetComponent<ArticulationBody>();
        m_RbB = pendulumB.GetComponent<ArticulationBody>();
        m_TorqueA = m_TorqueB = Vector3.zero;
    }

    /// <summary>
    /// Resets the position and velocity of the agent and the goal.
    /// </summary>
    public void AgentReset()
    {
        pendulumA.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
        pendulumA.transform.rotation = Quaternion.Euler(180f, 0f, 0f);

        pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
        pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
    }

    public void FixedUpdate()
    {
        //float maxTorque = 1000f;
        float deltaTorque = 25f;

        m_TorqueA = Vector3.zero;
        
        if (Input.GetKey(KeyCode.A))
            m_TorqueA.x += deltaTorque;
        if (Input.GetKey(KeyCode.Z))
            m_TorqueA.x -= deltaTorque;
        if (Input.GetKey(KeyCode.S))
            m_TorqueA.y += deltaTorque;
        if (Input.GetKey(KeyCode.X))
            m_TorqueA.y -= deltaTorque;
        if (Input.GetKey(KeyCode.D))
            m_TorqueA.z += deltaTorque;
        if (Input.GetKey(KeyCode.C))
            m_TorqueA.z -= deltaTorque;

        //m_TorqueA.x = Mathf.Clamp(m_TorqueA.x, -1.0f, 1.0f);
        //m_TorqueA.y = Mathf.Clamp(m_TorqueA.y, -1.0f, 1.0f);
        //m_TorqueA.z = Mathf.Clamp(m_TorqueA.z, -1.0f, 1.0f);
        
        m_RbA.AddTorque(m_TorqueA);
        
        m_TorqueB = Vector3.zero;
        
        if (Input.GetKey(KeyCode.F))
            m_TorqueB.x += deltaTorque;
        if (Input.GetKey(KeyCode.V))
            m_TorqueB.x -= deltaTorque;
        if (Input.GetKey(KeyCode.G))
            m_TorqueB.y += deltaTorque;
        if (Input.GetKey(KeyCode.B))
            m_TorqueB.y -= deltaTorque;
        if (Input.GetKey(KeyCode.H))
            m_TorqueB.z += deltaTorque;
        if (Input.GetKey(KeyCode.N))
            m_TorqueB.z -= deltaTorque;

        m_RbB.AddTorque(m_TorqueB);
    
        
        if (Input.GetKey(KeyCode.Escape))
            AgentReset();
    }
};

