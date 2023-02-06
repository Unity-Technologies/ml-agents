using System.Collections;
using System.Collections.Generic;
using RootMotion.Dynamics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class WalkerASEAgent : Agent
{
    public bool recordingMode;

    public Transform root;
    public Rigidbody rootRB;

    Vector3 m_OriginalPosition;
    Quaternion m_OriginalRotation;

    ConfigurableJointController m_Controller;

    public override void Initialize()
    {
        m_OriginalPosition = root.position;
        m_OriginalRotation = root.rotation;
        m_Controller = GetComponent<ConfigurableJointController>();
        if (!recordingMode)
        {
            var puppetMaster = GetComponentInChildren<PuppetMaster>();
            var animator = GetComponentInChildren<Animator>();
            Destroy(puppetMaster);
            Destroy(animator.gameObject);
        }
    }

    public override void OnEpisodeBegin()
    {
        ResetAgent();
        if (recordingMode)
        {
            ResetAnimation();
        }
    }
    void ResetAnimation()
    {
        // TODO reset animation on an episode reset (nice to have)
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

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        var continuousActions = actionBuffers.ContinuousActions;
        m_Controller.SetCJointTargets(continuousActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // add control from editor component
        for (int i = 0; i < m_Controller.cjControlSettings.Length; i++)
        {
            var target = m_Controller.cjControlSettings[i].target;
            var subIndex = 3 * i;
            continuousActionsOut[subIndex] = m_Controller.cjControlSettings[i].range.xRange.InverseScale(target.x);
            continuousActionsOut[++subIndex] = m_Controller.cjControlSettings[i].range.yRange.InverseScale(target.y);
            continuousActionsOut[++subIndex] = m_Controller.cjControlSettings[i].range.zRange.InverseScale(target.z);
        }

    }

    float GetRootHeightFromGround()
    {
        int layerMask = 1 << 3;
        layerMask = ~layerMask;
        Physics.Raycast(root.transform.position, Vector3.down, out var raycastHit, 10, layerMask);
        return raycastHit.distance;
    }

    float GetRootBalance()
    {
        var agentUp = root.transform.TransformDirection(Vector3.up);
        return Vector3.Dot(agentUp, Vector3.up);
    }

    Vector3 GetVelocity()
    {
        return rootRB.velocity;
    }

    Vector3 GetAngularVelocity()
    {
        return rootRB.angularVelocity;
    }
}
