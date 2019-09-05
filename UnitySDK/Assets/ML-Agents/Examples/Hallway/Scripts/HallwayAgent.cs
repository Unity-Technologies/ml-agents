using System.Collections;
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
    RayPerception m_RayPer;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    HallwayAcademy m_Academy;
    int m_Selection;

    protected override void InitializeAgent()
    {
        base.InitializeAgent();
        m_Academy = FindObjectOfType<HallwayAcademy>();
        m_RayPer = GetComponent<RayPerception>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
    }

    protected override void CollectObservations()
    {
        if (useVectorObs)
        {
            float rayDistance = 12f;
            float[] rayAngles = { 20f, 60f, 90f, 120f, 160f };
            string[] detectableObjects = { "orangeGoal", "redGoal", "orangeBlock", "redBlock", "wall" };
            AddVectorObs(GetStepCount() / (float)agentParameters.maxStep);
            AddVectorObs(m_RayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
        }
    }

    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        // ReSharper disable once Unity.InefficientPropertyAccess
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(float[] act)
    {
        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        if (brain.brainParameters.vectorActionSpaceType == SpaceType.Continuous)
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
        m_AgentRb.AddForce(dirToGo * m_Academy.agentRunSpeed, ForceMode.VelocityChange);
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
            if ((m_Selection == 0 && col.gameObject.CompareTag("orangeGoal")) ||
                (m_Selection == 1 && col.gameObject.CompareTag("redGoal")))
            {
                SetReward(1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_Academy.goalScoredMaterial, 0.5f));
            }
            else
            {
                SetReward(-0.1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_Academy.failMaterial, 0.5f));
            }
            Done();
        }
    }

    public override void AgentReset()
    {
        float agentOffset = -15f;
        float blockOffset = 0f;
        m_Selection = Random.Range(0, 2);
        var position = ground.transform.position;
        if (m_Selection == 0)
        {
            orangeBlock.transform.position =
                new Vector3(0f + Random.Range(-3f, 3f), 2f, blockOffset + Random.Range(-5f, 5f))
                + position;
            redBlock.transform.position =
                new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                + position;
        }
        else
        {
            orangeBlock.transform.position =
                new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                + position;
            redBlock.transform.position =
                new Vector3(0f, 2f, blockOffset + Random.Range(-5f, 5f))
                + position;
        }

        transform.position = new Vector3(0f + Random.Range(-3f, 3f),
            1f, agentOffset + Random.Range(-5f, 5f))
            + ground.transform.position;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        m_AgentRb.velocity *= 0f;

        int goalPos = Random.Range(0, 2);
        var areaPos = area.transform.position;
        if (goalPos == 0)
        {
            orangeGoal.transform.position = new Vector3(7f, 0.5f, 9f) + areaPos;
            redGoal.transform.position = new Vector3(-7f, 0.5f, 9f) + areaPos;
        }
        else
        {
            redGoal.transform.position = new Vector3(7f, 0.5f, 9f) + areaPos;
            orangeGoal.transform.position = new Vector3(-7f, 0.5f, 9f) + areaPos;
        }
    }
}
