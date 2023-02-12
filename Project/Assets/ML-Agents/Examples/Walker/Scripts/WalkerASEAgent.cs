using System;
using System.Linq;
using RootMotion.Dynamics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

public class WalkerASEAgent : Agent
{
    public bool recordingMode;

    public Transform root;
    public Transform chest;
    public Transform leftHand;
    public Transform rightHand;
    public Transform leftFoot;
    public Transform rightFoot;
    public Rigidbody rootRB;
    [Range(0f, 1f)]
    public float randomDropProbability;

    [Range(0f, 1f)]
    public float randomStandProbability;

    [Range(0f, 5f)]
    public float minSpawnHeight = 2f;

    [Range(0f, 5f)]
    public float maxSpawnHeight = 4f;

    public float StartHeight => m_StartingHeight;
    public int DecisionPeriod => m_DecisionPeriod;

    public float HeadTerminationHeight = 0.3f;
    public float BodyPartTerminationHeight = 0.15f;

    Vector3 m_OriginalPosition;
    Quaternion m_OriginalRotation;

    ConfigurableJointController m_Controller;
    LatentRequestor m_LatentRequestor;
    private float m_StartingHeight;
    private int m_DecisionPeriod;
    LocalFrameController m_FrameController;
    string[] m_SingleAxis = new[] { "shinL", "shinR", "lower_arm_L", "lower_arm_R" };
    string[] m_DualAxis = new[] { "hand_L", "hand_R" };
    bool m_IsRecoveryEpisode = false;


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
            var demoRecorder = GetComponent<DemonstrationRecorder>();
            var demoNamer = GetComponent<SetDemoNameToAnimationName>();
            Destroy(puppetMaster);
            Destroy(animator.gameObject);
            Destroy(demoRecorder);
            Destroy(demoNamer);
        }

        m_StartingHeight = GetRootHeightFromGround();
        m_DecisionPeriod = GetComponent<DecisionRequester>().DecisionPeriod;
        m_FrameController = GetComponentInChildren<LocalFrameController>();
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

        m_FrameController.UpdateLocalFrame(root);
    }

    void FixedUpdate()
    {
        m_FrameController.UpdateLocalFrame(root);
        if (CheckEpisodeTermination())
        {
            EndEpisode();
        }
    }
    bool CheckEpisodeTermination()
    {
        if (!m_IsRecoveryEpisode)
        {
            // check root
            var position = m_Controller.ConfigurableJointChain[0].transform.position;

            if (position.y < BodyPartTerminationHeight)
            {
                return true;
            }

            for (int i = 0; i < m_Controller.cjControlSettings.Length; i++)
            {
                var name = m_Controller.cjControlSettings[i].name;
                position = m_Controller.ConfigurableJointChain[i + 1].transform.position;
                if (name == "head")
                {
                    if (position.y < HeadTerminationHeight)
                    {
                        return true;
                    }
                }
                else
                {
                    if (position.y < BodyPartTerminationHeight && !(name == "footR" || name == "footL"))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }
    void ResetAnimation()
    {
        // TODO reset animation on an episode reset (nice to have)
    }

    void ResetAgent()
    {
        float[] angles = new float[m_Controller.cjControlSettings.Length * 3];
        for (int i = 0; i < m_Controller.cjControlSettings.Length; i++)
        {
            angles[i] = Random.Range(-1f, 1f);
        }

        var rand = Random.Range(0f, 1f);

        m_IsRecoveryEpisode = false;

        if (rand <= randomDropProbability)
        {
            var pos = GetRandomSpawnPosition(minSpawnHeight, maxSpawnHeight);
            var rot = GetRandomRotation();
            m_IsRecoveryEpisode = true;
            StartCoroutine(m_Controller.ResetCJointTargetsAndPositions(pos, rot, true));
        }
        else if (rand > randomDropProbability && rand <= randomStandProbability + randomDropProbability)
        {

            // m_Controller.SetPosRot(m_OriginalPosition, m_OriginalRotation);
            StartCoroutine(m_Controller.ResetCJointTargetsAndPositions(m_OriginalPosition, m_OriginalRotation, false));
        }
        else
        {
            // TODO reset to original position with randomized joint angles
        }

    }

    private Quaternion GetRandomRotation()
    {
        return Quaternion.Euler(Random.Range(0f, 360f), Random.Range(0f, 360f), Random.Range(0f, 360f));
    }

    private Vector3 GetRandomSpawnPosition(float yMin, float yMax)
    {
        var randomPosY = Random.Range(yMin, yMax);
        var randomSpawnPos = new Vector3(m_OriginalPosition.x, randomPosY, m_OriginalPosition.y);
        return randomSpawnPos;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(GetRootHeightFromGround());
        sensor.AddObservation(GetRootRotation());
        sensor.AddObservation(m_FrameController.transform.InverseTransformVector(GetVelocity()));
        sensor.AddObservation(m_FrameController.transform.InverseTransformVector(GetAngularVelocity()));
        sensor.AddObservation(GetRelativePosition(leftHand));
        sensor.AddObservation(GetRelativePosition(rightHand));
        sensor.AddObservation(GetRelativePosition(leftFoot));
        sensor.AddObservation(GetRelativePosition(rightFoot));
    }

    public override void CollectEmbedding(VectorSensor embedding)
    {
        // TODO Move this into the base agent script to avoid having to have the derived agent script do it.
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
        var offset = 0;
        for (int i = 0; i < m_Controller.cjControlSettings.Length; i++)
        {
            var target = m_Controller.cjControlSettings[i].target;
            var name = m_Controller.cjControlSettings[i].name;
            if (m_SingleAxis.Contains(name))
            {
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.xRange.InverseScale(target.x);
            }
            else if (m_DualAxis.Contains(name))
            {
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.xRange.InverseScale(target.x);
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.zRange.InverseScale(target.z);
            }
            else
            {
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.xRange.InverseScale(target.x);
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.yRange.InverseScale(target.y);
                continuousActionsOut[offset++] = m_Controller.cjControlSettings[i].range.zRange.InverseScale(target.z);
            }
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

    Quaternion GetRootRotation()
    {
        return Quaternion.FromToRotation(m_FrameController.transform.forward, root.forward);
    }

    Vector3 GetRelativePosition(Transform joint)
    {
        return m_FrameController.transform.InverseTransformPoint(joint.transform.position);
    }
}
