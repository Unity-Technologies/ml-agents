using UnityEngine;
using MLAgents;
using MLAgents.Sensors;

public class BasicAgent : MonoBehaviour
{
    public float timeBetweenDecisionsAtInference;
    float m_TimeSinceDecision;
    [HideInInspector]
    public int m_Position;
    const int k_SmallGoalPosition = 7;
    const int k_LargeGoalPosition = 17;
    public GameObject largeGoal;
    public GameObject smallGoal;
    const int k_MinPosition = 0;
    const int k_MaxPosition = 20;
    public const int k_Extents = k_MaxPosition - k_MinPosition;

    Agent m_Agent;

    public void OnEnable()
    {
        m_Agent = GetComponent<Agent>();
        ResetAgent();
    }

    public void ApplyAction(float[] vectorAction)
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
        if (m_Position < k_MinPosition) { m_Position = k_MinPosition; }
        if (m_Position > k_MaxPosition) { m_Position = k_MaxPosition; }

        gameObject.transform.position = new Vector3(m_Position - 10f, 0f, 0f);

        m_Agent.AddReward(-0.01f);

        if (m_Position == k_SmallGoalPosition)
        {
            m_Agent.AddReward(0.1f);
            m_Agent.Done();
            ResetAgent();
        }

        if (m_Position == k_LargeGoalPosition)
        {
            m_Agent.AddReward(1f);
            m_Agent.Done();
            ResetAgent();
        }
    }

    public void ResetAgent()
    {
        m_Position = 10;
        smallGoal.transform.position = new Vector3(k_SmallGoalPosition - 10f, 0f, 0f);
        largeGoal.transform.position = new Vector3(k_LargeGoalPosition - 10f, 0f, 0f);
    }

//    public override float[] Heuristic()
//    {
//        if (Input.GetKey(KeyCode.D))
//        {
//            return new float[] { 2 };
//        }
//        if (Input.GetKey(KeyCode.A))
//        {
//            return new float[] { 1 };
//        }
//        return new float[] { 0 };
//    }

    public void FixedUpdate()
    {
        WaitTimeInference();
        ApplyAction(m_Agent.GetAction());
    }

    void WaitTimeInference()
    {
        if (!Academy.Instance.IsCommunicatorOn)
        {
            m_Agent.RequestDecision();
        }
        else
        {
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                m_TimeSinceDecision = 0f;
                m_Agent.RequestDecision();
            }
            else
            {
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
