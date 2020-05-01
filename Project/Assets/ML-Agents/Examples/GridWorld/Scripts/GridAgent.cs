using System;
using UnityEngine;
using System.Linq;
using Unity.MLAgents;
using UnityEngine.Serialization;

public class GridAgent : Agent
{
    [FormerlySerializedAs("m_Area")]
    [Header("Specific to GridWorld")]
    public GridArea area;
    public float timeBetweenDecisionsAtInference;
    float m_TimeSinceDecision;

    [Tooltip("Because we want an observation right before making a decision, we can force " +
        "a camera to render before making a decision. Place the agentCam here if using " +
        "RenderTexture as observations.")]
    public Camera renderCamera;

    [Tooltip("Selecting will turn on action masking. Note that a model trained with action " +
        "masking turned on may not behave optimally when action masking is turned off.")]
    public bool maskActions = true;

    const int k_NoAction = 0;  // do nothing!
    const int k_Up = 1;
    const int k_Down = 2;
    const int k_Left = 3;
    const int k_Right = 4;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public override void CollectDiscreteActionMasks(DiscreteActionMasker actionMasker)
    {
        // Mask the necessary actions if selected by the user.
        if (maskActions)
        {
           // Prevents the agent from picking an action that would make it collide with a wall
            var positionX = (int)transform.position.x;
            var positionZ = (int)transform.position.z;
            var maxPosition = (int)m_ResetParams.GetWithDefault("gridSize", 5f) - 1;

            if (positionX == 0)
            {
                actionMasker.SetMask(0, new []{ k_Left});
            }

            if (positionX == maxPosition)
            {
                actionMasker.SetMask(0, new []{k_Right});
            }

            if (positionZ == 0)
            {
                actionMasker.SetMask(0, new []{k_Down});
            }

            if (positionZ == maxPosition)
            {
                actionMasker.SetMask(0, new []{k_Up});
            }
        }
    }

    // to be implemented by the developer
    public override void OnActionReceived(float[] vectorAction)
    {
        AddReward(-0.01f);
        var action = Mathf.FloorToInt(vectorAction[0]);

        var targetPos = transform.position;
        switch (action)
        {
            case k_NoAction:
                // do nothing
                break;
            case k_Right:
                targetPos = transform.position + new Vector3(1f, 0, 0f);
                break;
            case k_Left:
                targetPos = transform.position + new Vector3(-1f, 0, 0f);
                break;
            case k_Up:
                targetPos = transform.position + new Vector3(0f, 0, 1f);
                break;
            case k_Down:
                targetPos = transform.position + new Vector3(0f, 0, -1f);
                break;
            default:
                throw new ArgumentException("Invalid action value");
        }

        var hit = Physics.OverlapBox(
            targetPos, new Vector3(0.3f, 0.3f, 0.3f));
        if (hit.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length == 0)
        {
            transform.position = targetPos;

            if (hit.Where(col => col.gameObject.CompareTag("goal")).ToArray().Length == 1)
            {
                SetReward(1f);
                EndEpisode();
            }
            else if (hit.Where(col => col.gameObject.CompareTag("pit")).ToArray().Length == 1)
            {
                SetReward(-1f);
                EndEpisode();
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = k_NoAction;
        if (Input.GetKey(KeyCode.D))
        {
            actionsOut[0] = k_Right;
        }
        if (Input.GetKey(KeyCode.W))
        {
            actionsOut[0] = k_Up;
        }
        if (Input.GetKey(KeyCode.A))
        {
            actionsOut[0] = k_Left;
        }
        if (Input.GetKey(KeyCode.S))
        {
            actionsOut[0] = k_Down;
        }
    }

    // to be implemented by the developer
    public override void OnEpisodeBegin()
    {
        area.AreaReset();
    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    void WaitTimeInference()
    {
        if (renderCamera != null)
        {
            renderCamera.Render();
        }

        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                m_TimeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
