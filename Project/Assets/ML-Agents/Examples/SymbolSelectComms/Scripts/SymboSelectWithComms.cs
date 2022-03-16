//Put this script on your blue cube.

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

public class SymboSelectWithComms : Agent
{

    private PushBlockSettings m_PushBlockSettings;
    private Rigidbody m_AgentRb;  //cached on initialization
    private SymbolSelectWithCommsEnvController envController;
    private VectorSensorComponent commSensor;
    // public float[] message0 = new float[]{0,0};
    public float[] message0;
    public int assignedNumber;
    // public float[] assignedNumber = new float[]{0,0}; //the tile/number randomly assigned to an agent during an episode

    public Transform[] targetArray; //array of possible targets
    public int answerChoice = -1; //the tile/number randomly assigned to an agent during an episode
    // public int answerChoice1 = -1; //the tile/number randomly assigned to an agent during an episode
    public bool hasMovedToTarget;
    private Vector3 m_StartingPos;
    private Quaternion m_StartingRot;
    public float stepCounter;

    public int NumOfMessageBits = 4;
    // private float[] zeroOneHot = new float[] { 1, 0 };
    // private float[] oneOneHot = new float[] { 0, 1 };

    // private float[] zeroOneHot = new float[] {1,0,0,0};
    // private float[] oneOneHot = new float[] { 0,1,0,0};
    // private float[] twoOneHot = new float[] { 0,0,1,0};
    // private float[] threeOneHot = new float[] { 0,0,0,1};
    public override void Initialize()
    {
        base.Initialize();
        // zeroOneHot = new float[] { 1, 0 };
        // oneOneHot = new float[] { 0, 1 };
        message0 = new float[NumOfMessageBits];
        // GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.BranchSizes = new[] { 1, NumOfMessageBits };
        // message0 = new float[] { 0, 0, 0, 0};
        // zeroOneHot = new float[] {1,0,0,0};
        // oneOneHot = new float[] {0,1,0,0};
        // twoOneHot = new float[] { 0,0,1,0};
        // threeOneHot = new float[] { 0,0,0,1};
        m_AgentRb = GetComponent<Rigidbody>();
        envController = GetComponentInParent<SymbolSelectWithCommsEnvController>();
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        commSensor = GetComponent<VectorSensorComponent>();
        m_StartingPos = transform.position;
        m_StartingRot = transform.rotation;

    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        ResetAgent();
    }

    public float[] selectionKeys;

    void ResetAgent()
    {

        answerChoice = -1;
        assignedNumber = Random.Range(0, NumOfMessageBits);
        // message0 = new float[] { 0, 0 };
        message0 = new float[NumOfMessageBits];
        selectionKeys = new float[NumOfMessageBits];
        for (int i = 0; i < NumOfMessageBits; i++)
        {
            var t = i / NumOfMessageBits;
            selectionKeys[i] = Mathf.Lerp(-1, 1, t);
        }

        hasMovedToTarget = false;
        transform.SetPositionAndRotation(m_StartingPos, m_StartingRot);
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;
        stepCounter = 0;
        StartCoroutine(MoveToTarget());
    }

    // /// <summary>
    // /// Moves the agent according to the selected action.
    // /// </summary>
    // // public void MoveAgent(ActionSegment<int> act)
    // public void MoveAgent(ActionBuffers act)
    // {
    //     var continuousActions = act.ContinuousActions;
    //     // var discreteActions = act.DiscreteActions;
    //
    //
    //
    //     //ADD FORCE
    //     var inputV = continuousActions[0];
    //     var inputH = continuousActions[1];
    //     var rotate = continuousActions[2];
    //
    //     var moveDir = transform.TransformDirection(new Vector3(inputH, 0, inputV));
    //     Quaternion rotAmount = Quaternion.AngleAxis(Time.fixedDeltaTime * 200f * rotate, Vector3.up);
    //     m_AgentRb.MoveRotation(m_AgentRb.rotation * rotAmount);
    //     m_AgentRb.AddForce(moveDir * m_PushBlockSettings.agentRunSpeed,
    //         ForceMode.VelocityChange);
    // }

    // private void OnCollisionEnter(Collision other)
    // {
    //     if (other.gameObject.CompareTag("tile"))
    //     {
    //
    //     }
    // }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(stepCounter); //2bit one-hot
        stepCounter += .33f;
        var assignedOneHot = new float[NumOfMessageBits];
        assignedOneHot[assignedNumber] = 1;
        // var assignedOneHot=
        //     assignedNumber == 0 ? zeroOneHot :
        //     assignedNumber == 1 ? oneOneHot :
        //     assignedNumber == 2 ? twoOneHot : threeOneHot;
        sensor.AddObservation(assignedOneHot); //
        foreach (var item in envController.AgentsArray)
        {
            if (item.Agent != this)
            {
                commSensor.GetSensor().AddObservation(item.Agent.message0);
            }
        }
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var contAct = actionBuffers.ContinuousActions[0];
        // var selection1 = actionBuffers.ContinuousActions[1];
        // answerChoice = selection < 0? 0: 1; //if negative set to zero, positive set to 1
        // Mathf.Approximately()
        var selRange = ((contAct + 1) * .5f) * NumOfMessageBits;
        answerChoice = (int)selRange;

        // var zero = -1 < selection && selection < -.5f;
        // var one = -.5f < selection && selection < 0f;
        // var two = 0 < selection && selection < 0.5f;
        // var three = .5f < selection && selection < 1f;

        // answerChoice = zero? 0 : one? 1: two? 2: 3; //if negative set to zero, positive set to 1
        // answerChoice = selection0 < 0? 0: 1; //if negative set to zero, positive set to 1
        var messageBranch = actionBuffers.DiscreteActions[0];
        // var word0 = new float[]{0,0};
        // var word0 = new float[]{0,0,0,0};
        var word0 = new float[NumOfMessageBits];
        word0[messageBranch] = 1;
        message0 = word0;
    }


    //MOVE OVER THE COURSE OF 2 FIXEDUPDATE TICS
    IEnumerator MoveToTarget()
    {
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        for (int i = 0; i < 2; i++)
        {
            if (Academy.Instance.IsCommunicatorOn) { RequestDecision(); }
            yield return wait;
        }
        //make some decisions
        // if (Academy.Instance.IsCommunicatorOn){ RequestDecision(); }
        // yield return wait;
        // if (Academy.Instance.IsCommunicatorOn){ RequestDecision(); }
        // yield return wait;
        // if (Academy.Instance.IsCommunicatorOn){ RequestDecision(); }
        // yield return wait;
        // if (Academy.Instance.IsCommunicatorOn){ RequestDecision(); }
        // yield return wait;
        //
        //
        // // m_AgentRb.MovePosition(targetArray[answerChoice].position);
        // yield return wait;
        hasMovedToTarget = true;
    }

    // public override void Heuristic(in ActionBuffers actionsOut)
    // {
    //     // if (Input.GetKey(KeyCode.D))
    //     // {
    //     // print("heuristic");
    //     // }
    //     var acts = actionsOut.DiscreteActions;
    //     if (canChooseNow)
    //     {
    //         //FIRST AGENT WILL CHOOSE RANDOMLY
    //         if (isFirstOne)
    //         {
    //             canChooseNow = false;
    //             hasChosen = true;
    //
    //             acts[0] = Random.Range(0,2); //SELECTION BRANCH
    //             acts[1] = acts[0]; //MESSAGE IS SAME INDEX AS SELECTION
    //             //SECOND AGENT CAN NOW CHOOSE
    //             envController.AgentsArray[1].Agent.canChooseNow = true;
    //         }
    //     }
    // }
}
