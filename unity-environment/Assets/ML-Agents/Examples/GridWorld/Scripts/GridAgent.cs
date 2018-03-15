using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.UI;
using System.Linq;
using Newtonsoft.Json;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class GridAgent : Agent
{
    [Header("Specific to GridWorld")]
    private GridAcademy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(GridAcademy)) as GridAcademy;
    }

    public override void CollectObservations()
    {

    }

    // to be implemented by the developer
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-0.01f);
        int action = Mathf.FloorToInt(vectorAction[0]);

        // 0 - Forward, 1 - Backward, 2 - Left, 3 - Right
        Vector3 targetPos = transform.position;
        if (action == 3)
        {
            targetPos = transform.position + new Vector3(1f, 0, 0f);
        }

        if (action == 2)
        {
            targetPos = transform.position + new Vector3(-1f, 0, 0f);
        }

        if (action == 0)
        {
            targetPos = transform.position + new Vector3(0f, 0, 1f);
        }

        if (action == 1)
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
