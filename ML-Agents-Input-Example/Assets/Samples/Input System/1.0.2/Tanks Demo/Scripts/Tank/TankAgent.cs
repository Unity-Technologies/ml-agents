using System;
using System.Net.Sockets;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class TankAgent : Agent
{
    TankShooting m_Shooting;
    BehaviorParameters m_Bp;
    Rigidbody m_Rb;

    void Awake()
    {
        m_Bp = GetComponent<BehaviorParameters>();
        m_Shooting = GetComponent<TankShooting>();
        m_Rb = GetComponent<Rigidbody>();
    }

    public override void Initialize()
    {
    }

    public void SetTeam(int team)
    {
        m_Bp.TeamId = team;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Vector to opponent
        sensor.AddObservation(transform.localRotation.normalized.y);
        var velocity = transform.InverseTransformDirection(m_Rb.velocity);
        // charge time
        sensor.AddObservation(m_Shooting.m_CurrentLaunchForce / m_Shooting.maxLaunchForce);
        sensor.AddObservation(velocity.x);
        sensor.AddObservation(velocity.z);
    }
}
