using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class HallwayCollabAgent : HallwayAgent
{
    public GameObject symbolSGoal;
    public GameObject symbolS;
    public HallwayCollabAgent teammate;
    public bool isSpotter = true;
    TextMesh m_MessageText;
    TextMesh m_MessageRec;
    int m_Message = 0;


    [HideInInspector]
    public int selection = 0;

    public override void Initialize()
    {
        base.Initialize();
        if (isSpotter)
        {
            m_MessageText = gameObject.GetComponentInChildren<TextMesh>();
        }

    }

    public override void OnEpisodeBegin()
    {
        m_Message = -1;
        var agentOffset = 10f;
        if (isSpotter)
        {
            agentOffset = -15;
        }

        if (!isSpotter)
        {
            transform.position = new Vector3(0f + Random.Range(-3f, 3f),
                1f, agentOffset + Random.Range(-5f, 5f))
                + ground.transform.position;
            transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        }
        else
        {
            transform.position = new Vector3(0f,
                1f, agentOffset)
                + ground.transform.position;
            transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        }

        // Remove the randomness

        m_AgentRb.velocity *= 0f;
        if (isSpotter)
        {
            var blockOffset = -9f;
            // Only the Spotter has the correct selection
            selection = Random.Range(0, 3);
            if (selection == 0)
            {
                symbolO.transform.position =
                    new Vector3(0f, 2f, blockOffset)
                    + ground.transform.position;
                symbolX.transform.position =
                    new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                    + ground.transform.position;
                symbolS.transform.position =
                    new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                    + ground.transform.position;
            }
            else if (selection == 1)
            {
                symbolO.transform.position =
                    new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                    + ground.transform.position;
                symbolX.transform.position =
                    new Vector3(0f, 2f, blockOffset)
                    + ground.transform.position;
                symbolS.transform.position =
                    new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                    + ground.transform.position;
            }
            else
            {
                symbolO.transform.position =
                    new Vector3(0f, -1000f, blockOffset + Random.Range(-5f, 5f))
                    + ground.transform.position;
                symbolX.transform.position =
                    new Vector3(0f, -1000f, blockOffset)
                    + ground.transform.position;
                symbolS.transform.position =
                    new Vector3(0f, 2f, blockOffset)
                    + ground.transform.position;
            }

            var goalPos = Random.Range(0, 7);
            if (goalPos == 0)
            {
                symbolOGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
            }
            else if (goalPos == 1)
            {
                symbolOGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
            }
            else if (goalPos == 2)
            {
                symbolOGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
            }
            else if (goalPos == 3)
            {
                symbolOGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
            }
            else if (goalPos == 4)
            {
                symbolOGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
            }
            else
            {
                symbolOGoal.transform.position = new Vector3(0f, 0.5f, 22.29f) + area.transform.position;
                symbolXGoal.transform.position = new Vector3(7f, 0.5f, 22.29f) + area.transform.position;
                symbolSGoal.transform.position = new Vector3(-7f, 0.5f, 22.29f) + area.transform.position;
            }
        }
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        //if (useVectorObs)
        //{
        //    sensor.AddObservation(StepCount / (float)MaxStep);
        //}
        sensor.AddObservation(toOnehot(m_Message));
    }

    float[] toOnehot(int message)
    {
        float[] onehot = new float[3];
        if (message < 0 || message >= 3)
        {
            return onehot;
        }
        onehot[message] = 1f;
        return onehot;
    }

    public void tellAgent(int message)
    {
        m_Message = message;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        AddReward(-1f / MaxStep);
        if (!isSpotter)
        {
            MoveAgent(actionBuffers.DiscreteActions);
        }

        int comm_act = actionBuffers.DiscreteActions[1];

        if (isSpotter)
        {
            m_MessageText.text = "Sent:" + comm_act.ToString();
        }
        teammate.tellAgent(comm_act);
        // if (isSpotter) // Test
        // {
        //     teammate.tellAgent(selection);
        // }
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("symbol_O_Goal") || col.gameObject.CompareTag("symbol_X_Goal") || col.gameObject.CompareTag("symbol_S_Goal"))
        {
            if (!isSpotter)
            {
                // Check the ground truth
                if ((teammate.selection == 0 && col.gameObject.CompareTag("symbol_O_Goal")) ||
                    (teammate.selection == 1 && col.gameObject.CompareTag("symbol_X_Goal")) ||
                    (teammate.selection == 2 && col.gameObject.CompareTag("symbol_S_Goal")))
                {
                    SetReward(1f);
                    teammate.SetReward(1f);
                    StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.goalScoredMaterial, 0.5f));
                }
                else
                {
                    SetReward(-0.1f);
                    teammate.SetReward(-0.1f);
                    StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.failMaterial, 0.5f));
                }
                EndEpisode();
                teammate.EndEpisode();
            }
        }
    }

    //public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    //{
    //    // Mask the necessary actions if selected by the user.
    //    if (!isSpotter)
    //    {
    //        // Prevents the agent from picking an action that would make it collide with a wall
    //        actionMask.WriteMask(1, new[] { 0 });
    //    }
    //}

}
