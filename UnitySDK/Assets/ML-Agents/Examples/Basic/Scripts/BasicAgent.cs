using UnityEngine;
using MLAgents;

public class BasicAgent : Agent
{
    [Header("Specific to Basic")]
    BasicAcademy m_Academy;
    public float timeBetweenDecisionsAtInference;
    float m_TimeSinceDecision;
    int m_Position;
    int m_SmallGoalPosition;
    int m_LargeGoalPosition;
    public GameObject largeGoal;
    public GameObject smallGoal;
    int m_MinPosition;
    int m_MaxPosition;

    public override void InitializeAgent()
    {
        m_Academy = FindObjectOfType(typeof(BasicAcademy)) as BasicAcademy;
    }

    public override void CollectObservations()
    {
        AddVectorObs(m_Position, 20);
    }

    public override void AgentAction(float[] vectorAction)
    {
        var movement = (int)vectorAction[0];

        var direction = 0;

        switch (movement)
        {
            case 1:
                direction = -1;
                break;
            case 2:
                direction = 1;
                break;
        }

        m_Position += direction;
        if (m_Position < m_MinPosition) { m_Position = m_MinPosition; }
        if (m_Position > m_MaxPosition) { m_Position = m_MaxPosition; }

        gameObject.transform.position = new Vector3(m_Position - 10f, 0f, 0f);

        AddReward(-0.01f);

        if (m_Position == m_SmallGoalPosition)
        {
            Done();
            AddReward(0.1f);
        }

        if (m_Position == m_LargeGoalPosition)
        {
            Done();
            AddReward(1f);
        }
    }

    public override void AgentReset()
    {
        m_Position = 10;
        m_MinPosition = 0;
        m_MaxPosition = 20;
        m_SmallGoalPosition = 7;
        m_LargeGoalPosition = 17;
        smallGoal.transform.position = new Vector3(m_SmallGoalPosition - 10f, 0f, 0f);
        largeGoal.transform.position = new Vector3(m_LargeGoalPosition - 10f, 0f, 0f);
    }

    public override float[] Heuristic()
    {
        if (Input.GetKey(KeyCode.D))
        {
            return new float[] { 2 };
        }
        if (Input.GetKey(KeyCode.A))
        {
            return new float[] { 1 };
        }
        return new float[] { 0 };
    }

    public override void AgentOnDone()
    {
    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    void WaitTimeInference()
    {
        if (!m_Academy.GetIsInference())
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
