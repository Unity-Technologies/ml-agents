using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class TennisArea : MonoBehaviour {

    public static Queue<char> last1000BrainResults = new Queue<char>();
    public static Queue<char> last1000AgentResults = new Queue<char>();
    public static Queue<int> last1000Passes = new Queue<int>();

    public static void AddPasses(int nbPasses){
        last1000Passes.Enqueue(nbPasses);
        if (last1000Passes.Count > 1000){
            last1000Passes.Dequeue();
        }
    }

    public static void AddResult(bool isLearnWinner, bool isAgentAWinner){
        // Debug.Log("Add result learnIsWinner " + isLearnWinner + " AisWinner: " + isAgentAWinner);

        if (isLearnWinner)
            learningScore ++;
        else
            ghostScore++;

        if (isAgentAWinner)
            agentAScore++;
        else  
            agentBScore++;

        last1000BrainResults.Enqueue(isLearnWinner ? 'L' : 'G');
        if (last1000BrainResults.Count > 1000){
            last1000BrainResults.Dequeue();
        }

        last1000AgentResults.Enqueue(isAgentAWinner ? 'A' : 'B');
        if (last1000AgentResults.Count > 1000){
            last1000AgentResults.Dequeue();
        }
    }
    public static int agentAScore;
    public static int agentBScore;
    public static int ghostScore;
    public static int learningScore;
    public Brain learningBrain;
    public Brain ghostBrain;

    public GameObject ball;
    public TennisAgent agentA;
    public TennisAgent agentB;
    private Rigidbody ballRb;

    private bool agentAIsLearning;

    void Start ()
    {
        ballRb = ball.GetComponent<Rigidbody>();
        MatchReset();
    }
    
    public void MatchReset() 
    {
        float ballOut = Random.Range(6f, 8f);
        int flip = Random.Range(0, 2);
        if (flip == 0)
        {
            ball.transform.position = new Vector3(-ballOut, 6f, 0f) + transform.position;
        }
        else
        {
            ball.transform.position = new Vector3(ballOut, 6f, 0f) + transform.position;
        }
        ballRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.localScale = new Vector3(1, 1, 1);
        ball.GetComponent<HitWall>().lastAgentHit = -1;
        bool agentAIsLearning = Random.Range(0,2) == 0f;
        Brain agentABrain = agentAIsLearning ? learningBrain : ghostBrain;
        Brain agentBBrain = agentAIsLearning ? ghostBrain : learningBrain;
        agentA.SetBrain(agentABrain, agentAIsLearning);
        agentB.SetBrain(agentBBrain, !agentAIsLearning);
    }

    void FixedUpdate() 
    {
        Vector3 rgV = ballRb.velocity;
        ballRb.velocity = new Vector3(Mathf.Clamp(rgV.x, -9f, 9f), Mathf.Clamp(rgV.y, -9f, 9f), rgV.z);
    }
}
