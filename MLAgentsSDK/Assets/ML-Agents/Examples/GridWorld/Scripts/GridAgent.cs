using UnityEngine;
using System.Linq;
using MLAgents;

public class GridAgent : Agent
{
    [Header("Specific to GridWorld")]
    private GridAcademy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;

    static int up = 1;
    static int down = 2;
    static int left = 3;
    static int right = 4;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(GridAcademy)) as GridAcademy;
    }

    public override void CollectObservations()
    {
        // Prevents the agent from picking an action that would make it collide with a wall
        var positionX = (int) transform.position.x;
        var positionZ = (int) transform.position.z;
        var maxPosition = academy.gridSize - 1;
        if (positionX == 0)
        {
            SetActionMask(left);
        }
        if (positionX == maxPosition)
        {
            SetActionMask(right);
        }
        if (positionZ == 0)
        {
            SetActionMask(down);
        }
        if (positionZ == maxPosition)
        {
            SetActionMask(up);
        }
    }

    // to be implemented by the developer
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-0.01f);
        int action = Mathf.FloorToInt(vectorAction[0]);

        Vector3 targetPos = transform.position;
        if (action == right)
        {
            targetPos = transform.position + new Vector3(1f, 0, 0f);
        }

        if (action == left)
        {
            targetPos = transform.position + new Vector3(-1f, 0, 0f);
        }

        if (action == up)
        {
            targetPos = transform.position + new Vector3(0f, 0, 1f);
        }

        if (action == down)
        {
            targetPos = transform.position + new Vector3(0f, 0, -1f);
        }

        Collider[] blockTest = Physics.OverlapBox(targetPos, new Vector3(0.3f, 0.3f, 0.3f));
        if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
        {
            transform.position = targetPos;

            if (blockTest.Where(col => col.gameObject.tag == "goal").ToArray().Length == 1)
            {
                Done();
                SetReward(1f);
            }
            if (blockTest.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
            {
                Done();
                SetReward(-1f);
            }

        }
    }

    // to be implemented by the developer
    public override void AgentReset()
    {
        academy.AcademyReset();
    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    private void WaitTimeInference()
    {
        if (!academy.GetIsInference())
        {
            RequestDecision();
        }
        else
        {
            if (timeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                timeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                timeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
