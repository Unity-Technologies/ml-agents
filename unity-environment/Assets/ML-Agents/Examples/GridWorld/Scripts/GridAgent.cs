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
    public GridAcademy academy;

    public override void InitializeAgent()
    {

    }

    public override List<float> CollectState()
    {
        return state;
    }

    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {
        reward = -0.01f;
        int action = Mathf.FloorToInt(act[0]);

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
                done = true;
                reward = 1;
            }
            if (blockTest.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
            {
                done = true;
                reward = -1;
            }
        }
    }

    // to be implemented by the developer
    public override void AgentReset()
    {
        academy.AcademyReset();
    }
}
