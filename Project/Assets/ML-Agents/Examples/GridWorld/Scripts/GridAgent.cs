using System;
using UnityEngine;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Rendering;
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

    VectorSensorComponent m_GoalSensor;

    public enum GridGoal
    {
        GreenPlus,
        RedEx,
    }

    // Visual representations of the agent. Both are blue on top, but different colors on the bottom - this
    // allows the user to see which corresponds to the current goal, but it's not visible to the camera.
    // Only one is active at a time.
    public GameObject GreenBottom;
    public GameObject RedBottom;

    GridGoal m_CurrentGoal;

    public GridGoal CurrentGoal
    {
        get { return m_CurrentGoal; }
        set
        {
            switch (value)
            {
                case GridGoal.GreenPlus:
                    GreenBottom.SetActive(true);
                    RedBottom.SetActive(false);
                    break;
                case GridGoal.RedEx:
                    GreenBottom.SetActive(false);
                    RedBottom.SetActive(true);
                    break;
            }
            m_CurrentGoal = value;
        }
    }

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
        m_GoalSensor = this.GetComponent<VectorSensorComponent>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Array values = Enum.GetValues(typeof(GridGoal));

        if (m_GoalSensor is object)
        {
            int goalNum = (int)CurrentGoal;
            m_GoalSensor.GetSensor().AddOneHotObservation(goalNum, values.Length);
        }
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // Mask the necessary actions if selected by the user.
        if (maskActions)
        {
            // Prevents the agent from picking an action that would make it collide with a wall
            var positionX = (int)transform.localPosition.x;
            var positionZ = (int)transform.localPosition.z;
            var maxPosition = (int)m_ResetParams.GetWithDefault("gridSize", 5f) - 1;

            if (positionX == 0)
            {
                actionMask.SetActionEnabled(0, k_Left, false);
            }

            if (positionX == maxPosition)
            {
                actionMask.SetActionEnabled(0, k_Right, false);
            }

            if (positionZ == 0)
            {
                actionMask.SetActionEnabled(0, k_Down, false);
            }

            if (positionZ == maxPosition)
            {
                actionMask.SetActionEnabled(0, k_Up, false);
            }
        }
    }

    // to be implemented by the developer
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        AddReward(-0.01f);
        var action = actionBuffers.DiscreteActions[0];

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

            if (hit.Where(col => col.gameObject.CompareTag("plus")).ToArray().Length == 1)
            {
                ProvideReward(GridGoal.GreenPlus);
                EndEpisode();
            }
            else if (hit.Where(col => col.gameObject.CompareTag("ex")).ToArray().Length == 1)
            {
                ProvideReward(GridGoal.RedEx);
                EndEpisode();
            }
        }
    }

    private void ProvideReward(GridGoal hitObject)
    {
        if (CurrentGoal == hitObject)
        {
            SetReward(1f);
        }
        else
        {
            SetReward(-1f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = k_NoAction;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = k_Right;
        }
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = k_Up;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = k_Left;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = k_Down;
        }
    }

    // to be implemented by the developer
    public override void OnEpisodeBegin()
    {
        area.AreaReset();
        Array values = Enum.GetValues(typeof(GridGoal));
        if (m_GoalSensor is object)
        {
            CurrentGoal = (GridGoal)values.GetValue(UnityEngine.Random.Range(0, values.Length));
        }
        else
        {
            CurrentGoal = GridGoal.GreenPlus;
        }
    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    void WaitTimeInference()
    {
        if (renderCamera != null && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
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
