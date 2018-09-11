using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class HallwayAgent : Agent
{
    public GameObject ground;
    public GameObject area;
    public GameObject orangeGoal;
    public GameObject redGoal;
    public GameObject orangeBlock;
    public GameObject redBlock;
    public bool useVectorObs;
    RayPerception rayPer;
    Rigidbody shortBlockRB;
    Rigidbody agentRB;
    Material groundMaterial;
    Renderer groundRenderer;
    HallwayAcademy academy;
    int selection;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        academy = FindObjectOfType<HallwayAcademy>();
        rayPer = GetComponent<RayPerception>();
        agentRB = GetComponent<Rigidbody>();
        groundRenderer = ground.GetComponent<Renderer>();
        groundMaterial = groundRenderer.material;
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            float rayDistance = 12f;
            float[] rayAngles = { 20f, 60f, 90f, 120f, 160f };
            string[] detectableObjects = { "orangeGoal", "redGoal", "orangeBlock", "redBlock", "wall" };
            AddVectorObs(GetStepCount() / (float)agentParameters.maxStep);
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
        }
    }

    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        groundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        groundRenderer.material = groundMaterial;
    }

    public void MoveAgent(float[] act)
    {

        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
        }
        else
        {
            int action = Mathf.FloorToInt(act[0]);
            switch (action)
            {
                case 1:
                    dirToGo = transform.forward * 1f;
                    break;
                case 2:
                    dirToGo = transform.forward * -1f;
                    break;
                case 3:
                    rotateDir = transform.up * 1f;
                    break;
                case 4:
                    rotateDir = transform.up * -1f;
                    break;
            }
        }
        transform.Rotate(rotateDir, Time.deltaTime * 150f);
        agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-1f / agentParameters.maxStep);
        MoveAgent(vectorAction);
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("orangeGoal") || col.gameObject.CompareTag("redGoal"))
        {
            if ((selection == 0 && col.gameObject.CompareTag("orangeGoal")) ||
                (selection == 1 && col.gameObject.CompareTag("redGoal")))
            {
                SetReward(1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(academy.goalScoredMaterial, 0.5f));
            }
            else
            {
                SetReward(-0.1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, 0.5f));
            }
            Done();
        }
    }

    public override void AgentReset()
    {
        float agentOffset = -15f;
        float blockOffset = 0f;
        selection = Random.Range(0, 2);
        if (selection == 0)
        {
            orangeBlock.transform.position =
                new Vector3(0f + Random.Range(-3f, 3f), 2f, blockOffset + Random.Range(-5f, 5f))
                + ground.transform.position;
            redBlock.transform.position =
                new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                + ground.transform.position;
        }
        else
        {
            orangeBlock.transform.position =
                           new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                           + ground.transform.position;
            redBlock.transform.position =
                new Vector3(0f, 2f, blockOffset + Random.Range(-5f, 5f))
                + ground.transform.position;
        }

        transform.position = new Vector3(0f + Random.Range(-3f, 3f),
                                         1f, agentOffset + Random.Range(-5f, 5f))
            + ground.transform.position;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        agentRB.velocity *= 0f;

        int goalPos = Random.Range(0, 2);
        if (goalPos == 0)
        {
            orangeGoal.transform.position = new Vector3(7f, 0.5f, 9f) + area.transform.position;
            redGoal.transform.position = new Vector3(-7f, 0.5f, 9f) + area.transform.position;
        }
        else
        {
            redGoal.transform.position = new Vector3(7f, 0.5f, 9f) + area.transform.position;
            orangeGoal.transform.position = new Vector3(-7f, 0.5f, 9f) + area.transform.position;
        }
    }
}
