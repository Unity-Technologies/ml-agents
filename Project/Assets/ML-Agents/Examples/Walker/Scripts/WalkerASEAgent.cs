using RootMotion.Dynamics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class WalkerASEAgent : Agent
{
    public bool recordingMode;

    public Transform root;
    public Transform chest;
    public Rigidbody rootRB;

    public float StartHeight => m_StartingHeight;
    public int DecisionPeriod => m_DecisionPeriod;

    Vector3 m_OriginalPosition;
    Quaternion m_OriginalRotation;

    ConfigurableJointController m_Controller;
    LatentRequestor m_LatentRequestor;
    private float m_StartingHeight;
    private int m_DecisionPeriod;

    public override void Initialize()
    {
        m_OriginalPosition = root.localPosition;
        m_OriginalRotation = root.localRotation;
        m_LatentRequestor = GetComponent<LatentRequestor>();
        m_Controller = GetComponent<ConfigurableJointController>();
        if (!recordingMode)
        {
            var puppetMaster = GetComponentInChildren<PuppetMaster>();
            var animator = GetComponentInChildren<Animator>();
            Destroy(puppetMaster);
            Destroy(animator.gameObject);
        }

        m_StartingHeight = GetRootHeightFromGround();
        m_DecisionPeriod = GetComponent<DecisionRequester>().DecisionPeriod;
    }

    public override void OnEpisodeBegin()
    {
        ResetAgent();
        if (m_LatentRequestor != null)
        {
            m_LatentRequestor.ResetLatents();
            m_LatentRequestor.ResetLatentStepCounts();
        }
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
        root.localPosition = m_OriginalPosition;
        root.localRotation = m_OriginalRotation;
        StartCoroutine(m_Controller.ResetCJointTargetsAndPositions());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(GetRootHeightFromGround());
        sensor.AddObservation(GetChestBalance());
        sensor.AddObservation(GetRootBalance());
        sensor.AddObservation(root.up);
        sensor.AddObservation(root.forward);
        sensor.AddObservation(root.InverseTransformVector(GetVelocity()));
        sensor.AddObservation(root.InverseTransformVector(GetAngularVelocity()));
    }

    public override void CollectEmbedding(VectorSensor embedding)
    {
        // TODO Move this into the base agent script to avoid having to have the dervied agent script do it.
        if (embedding != null && embedding.ObservationSize() > 0)
        {
            if (Academy.Instance.IsCommunicatorOn || !recordingMode)
            {
                embedding.AddObservation(m_LatentRequestor.Latents);
            }
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        var continuousActions = actionBuffers.ContinuousActions;
        m_Controller.SetCJointTargets(continuousActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        for (int i = 0; i < m_Controller.cjControlSettings.Length; i++)
        {
            var target = m_Controller.cjControlSettings[i].target;
            var subIndex = 3 * i;
            continuousActionsOut[subIndex] = m_Controller.cjControlSettings[i].range.xRange.InverseScale(target.x);
            continuousActionsOut[++subIndex] = m_Controller.cjControlSettings[i].range.yRange.InverseScale(target.y);
            continuousActionsOut[++subIndex] = m_Controller.cjControlSettings[i].range.zRange.InverseScale(target.z);
        }

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

    public float GetChestBalance()
    {
        var agentUp = chest.transform.TransformDirection(Vector3.up);
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
