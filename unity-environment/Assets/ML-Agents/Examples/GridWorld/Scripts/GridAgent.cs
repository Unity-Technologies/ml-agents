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
    [HideInInspector]
    public int gridSize;

    GameObject trueAgent;

    public override void InitializeAgent()
    {
        trueAgent = gameObject;
        gridSize = (int)academy.resetParameters["gridSize"];
    }

    public override List<float> CollectState()
    {
        int closestGoalDistance = 2 * (int)academy.resetParameters["gridSize"];
        GameObject currentClosestGoal = academy.actorObjs[0];
        int closestPitDistance = 2 * (int)academy.resetParameters["gridSize"];
        GameObject currentClosestPit = academy.actorObjs[0];
        GameObject agent = academy.actorObjs[0];
        List<float> state = new List<float>();
        foreach (GameObject actor in academy.actorObjs)
        {
            if (actor.tag == "agent")
            {
                agent = actor;
                state.Add(actor.transform.position.x / (gridSize + 1));
                state.Add(actor.transform.position.z / (gridSize + 1));
                continue;
            }
        }
        foreach (GameObject actor in academy.actorObjs)
        {
            if (actor.tag == "goal")
            {
                int distance = (int)Mathf.Abs(agent.transform.position.x - actor.transform.position.x) + (int)Mathf.Abs(agent.transform.position.z - actor.transform.position.z);
                if (closestGoalDistance > distance)
                {
                    closestGoalDistance = distance;
                    currentClosestGoal = actor;
                }
            }
            if (actor.tag == "pit")
            {
                int distance = (int)Mathf.Abs(agent.transform.position.x - actor.transform.position.x) + (int)Mathf.Abs(agent.transform.position.z - actor.transform.position.z);
                if (closestPitDistance > distance)
                {
                    closestPitDistance = distance;
                    currentClosestPit = actor;
                }
            }
        }

        state.Add(currentClosestGoal.transform.position.x / (gridSize + 1));
        state.Add(currentClosestGoal.transform.position.z / (gridSize + 1));
        state.Add(currentClosestPit.transform.position.x / (gridSize + 1));
        state.Add(currentClosestPit.transform.position.z / (gridSize + 1));

        return state;
    }

    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {

        reward = -0.01f;
        int action = Mathf.FloorToInt(act[0]);

        // 0 - Forward, 1 - Backward, 2 - Left, 3 - Right
        if (action == 3)
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(trueAgent.transform.position.x + 1, 0, trueAgent.transform.position.z), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                trueAgent.transform.position = new Vector3(trueAgent.transform.position.x + 1, 0, trueAgent.transform.position.z);
            }
        }

        if (action == 2)
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(trueAgent.transform.position.x - 1, 0, trueAgent.transform.position.z), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                trueAgent.transform.position = new Vector3(trueAgent.transform.position.x - 1, 0, trueAgent.transform.position.z);
            }
        }

        if (action == 0)
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(trueAgent.transform.position.x, 0, trueAgent.transform.position.z + 1), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                trueAgent.transform.position = new Vector3(trueAgent.transform.position.x, 0, trueAgent.transform.position.z + 1);
            }
        }

        if (action == 1)
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(trueAgent.transform.position.x, 0, trueAgent.transform.position.z - 1), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                trueAgent.transform.position = new Vector3(trueAgent.transform.position.x, 0, trueAgent.transform.position.z - 1);
            }
        }

        Collider[] hitObjects = Physics.OverlapBox(trueAgent.transform.position, new Vector3(0.3f, 0.3f, 0.3f));
        if (hitObjects.Where(col => col.gameObject.tag == "goal").ToArray().Length == 1)
        {
            reward = 1f;
            done = true;
        }
        if (hitObjects.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
        {
            reward = -1f;
            done = true;
        }

        //if (trainMode == "train") {
        if (true)
        {
            academy.visualAgent.transform.position = trueAgent.transform.position;
            academy.visualAgent.transform.rotation = trueAgent.transform.rotation;
        }
    }

    // to be implemented by the developer
    public override void AgentReset()
    {
        academy.AcademyReset();
    }
}
