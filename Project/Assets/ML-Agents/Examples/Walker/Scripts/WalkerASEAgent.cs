using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class WalkerASEAgent : Agent
{
    public bool RecordingMode;

    public Transform root;
    public Rigidbody rootRB;

    Vector3 m_OriginalPosition;
    Quaternion m_OriginalRotation;

    public override void Initialize()
    {
        m_OriginalPosition = root.position;
        m_OriginalRotation = root.rotation;
    }

    public override void OnEpisodeBegin()
    {
        ResetAgent();
    }

    void ResetAgent()
    {
        root.position = m_OriginalPosition;
        root.rotation = m_OriginalRotation;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(GetRootHeightFromGround());
        sensor.AddObservation(GetRootBalance());
        sensor.AddObservation(root.up);
        sensor.AddObservation(root.forward);
        sensor.AddObservation(root.InverseTransformVector(GetVelocity()));
        sensor.AddObservation(root.InverseTransformVector(GetAngularVelocity()));
    }

    public override void CollectEmbedding(VectorSensor embedding)
    {

    }

    public float GetRootHeightFromGround()
    {
        int layerMask = 1 << 3;
        layerMask = ~layerMask;
        Physics.Raycast(root.transform.position, Vector3.down, out var raycastHit, 10, layerMask);
        return raycastHit.distance;
    }

    public float GetRootBalance()
    {
        var agentUp = root.transform.TransformDirection(Vector3.up);
        return Vector3.Dot(agentUp, Vector3.up);
    }

    public Vector3 GetVelocity()
    {
        return rootRB.velocity;
    }

    public Vector3 GetAngularVelocity()
    {
        return rootRB.angularVelocity;
    }
}
