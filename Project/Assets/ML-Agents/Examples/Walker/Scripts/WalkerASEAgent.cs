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
    public int RecoverySteps = 60;
    public bool EnableEarlyTermination = true;

    Vector3 m_OriginalPosition;
    Quaternion m_OriginalRotation;

    ConfigurableJointController m_Controller;
    LatentRequestor m_LatentRequestor;
    float m_StartingHeight;
    int m_DecisionPeriod;
    LocalFrameController m_AgentLocalFrameController;
    string[] m_SingleAxis = { "shinL", "shinR", "lower_arm_L", "lower_arm_R" };
    string[] m_DualAxis = { "hand_L", "hand_R" };
    bool m_IsRecoveryEpisode;
    int m_CurrentRecoverySteps = 0;

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
        m_AgentLocalFrameController = GetComponentInChildren<LocalFrameController>();
        Academy.Instance.AgentPreStep += IncrementRecoverySteps;
    }

    public override void OnEpisodeBegin()
    {
        // Reset walker state
        ResetAgent();

        // Reset latents if latent requestor present
        // TODO Do this automatically if latent requestor present and embedding size > 0
        if (m_LatentRequestor != null)
        {
            // reset latents and latent step count
            m_LatentRequestor.ResetLatents();
            m_LatentRequestor.ResetLatentStepCounts();
        }
        m_AgentLocalFrameController.UpdateLocalFrame(root);
    }

    void FixedUpdate()
    {
        m_AgentLocalFrameController.UpdateLocalFrame(root);

        if (CheckEpisodeTermination() && EnableEarlyTermination)
        {
            EndEpisode();
        }
    }
    bool CheckEpisodeTermination()
    {
        if (!m_IsRecoveryEpisode)
        {
            // check root for early termination
            var position = m_Controller.ConfigurableJointChain[0].transform.position;

            if (position.y < BodyPartTerminationHeight)
            {
                return true;
            }

            // check joints for early termination
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

    void IncrementRecoverySteps(int academyStepCount)
    {
        if (m_IsRecoveryEpisode)
        {
            if (m_CurrentRecoverySteps == RecoverySteps)
            {
                m_CurrentRecoverySteps = 0;
                EndEpisode();
            }
            else
            {
                m_CurrentRecoverySteps++;
            }
        }
    }

    void ResetAgent()
    {
        m_IsRecoveryEpisode = false;
        m_CurrentRecoverySteps = 0;

        if (!recordingMode)
        {
            var rand = Random.Range(0f, 1f);
            if (randomDropProbability + randomStandProbability > 1.0f)
            {
                throw new ArgumentException("Stand and drop probabilities must sum to 1.0.");
            }

            if (rand <= randomDropProbability)
            {
                var pos = GetRandomSpawnPosition(minSpawnHeight, maxSpawnHeight);
                var rot = GetRandomRotation();
                m_IsRecoveryEpisode = true;
                StartCoroutine(m_Controller.ResetCJointTargetsAndPositions(pos, Vector3.zero, rot, true));
            }
            else if (rand > randomDropProbability && rand <= randomStandProbability + randomDropProbability)
            {

                var verticalOffset = new Vector3(0f, 0.05f, 0f);
                StartCoroutine(m_Controller.ResetCJointTargetsAndPositions(m_OriginalPosition, verticalOffset, m_OriginalRotation, false));
            }
        }
        else
        {
            StartCoroutine(m_Controller.ResetCJointTargetsAndPositions(m_OriginalPosition, m_OriginalRotation));
        }
    }

    Quaternion GetRandomRotation()
    {
        return Quaternion.Euler(Random.Range(0f, 360f), Random.Range(0f, 360f), Random.Range(0f, 360f));
    }

    Vector3 GetRandomSpawnPosition(float yMin, float yMax)
    {
        var randomPosY = Random.Range(yMin, yMax);
        var randomSpawnPos = new Vector3(m_OriginalPosition.x, randomPosY, m_OriginalPosition.y);
        return randomSpawnPos;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(GetRootHeightFromGround()); // 1
        sensor.AddObservation(GetRelativeRootRotation()); // 4
        sensor.AddObservation(GetRelativeVelocity()); // 3
        sensor.AddObservation(GetRelativeAngularVelocity()); // 3
        sensor.AddObservation(GetRelativePosition(leftHand)); // 3
        sensor.AddObservation(GetRelativePosition(rightHand)); // 3
        sensor.AddObservation(GetRelativePosition(leftFoot)); // 3
        sensor.AddObservation(GetRelativePosition(rightFoot)); // 3
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

    Quaternion GetRelativeRootRotation()
    {
        return Quaternion.FromToRotation(m_AgentLocalFrameController.transform.forward, root.forward);
    }

    Vector3 GetRelativePosition(Transform joint)
    {
        return m_AgentLocalFrameController.transform.InverseTransformPoint(joint.transform.position);
    }

    Vector3 GetRelativeVelocity()
    {
        return m_AgentLocalFrameController.transform.InverseTransformDirection(GetVelocity());
    }

    Vector3 GetRelativeAngularVelocity()
    {
        return m_AgentLocalFrameController.transform.InverseTransformDirection(GetAngularVelocity());
    }
}
