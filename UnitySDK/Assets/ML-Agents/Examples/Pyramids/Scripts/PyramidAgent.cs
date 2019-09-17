using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;
using MLAgents;

public class PyramidAgent : Agent
{
    public GameObject area;
    private PyramidArea myArea;
    private Rigidbody agentRb;
    private RayPerception rayPer;
    private PyramidSwitch switchLogic;
    public GameObject areaSwitch;
    public bool useVectorObs;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        agentRb = GetComponent<Rigidbody>();
        myArea = area.GetComponent<PyramidArea>();
        rayPer = GetComponent<RayPerception>();
        switchLogic = areaSwitch.GetComponent<PyramidSwitch>();
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            const float rayDistance = 35f;
            float[] rayAngles = {20f, 90f, 160f, 45f, 135f, 70f, 110f};
            float[] rayAngles1 = {25f, 95f, 165f, 50f, 140f, 75f, 115f};
            float[] rayAngles2 = {15f, 85f, 155f, 40f, 130f, 65f, 105f};

            string[] detectableObjects = {"block", "wall", "goal", "switchOff", "switchOn", "stone"};
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles1, detectableObjects, 0f, 5f));
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles2, detectableObjects, 0f, 10f));
            AddVectorObs(switchLogic.GetState());
            AddVectorObs(transform.InverseTransformDirection(agentRb.velocity));
        }
    }

    public void MoveAgent(float[] act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
        }
        else
        {
            var action = Mathf.FloorToInt(act[0]);
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
        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        agentRb.AddForce(dirToGo * 2f, ForceMode.VelocityChange);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-1f / agentParameters.maxStep);
        MoveAgent(vectorAction);
    }

    public override void AgentReset()
    {
        var enumerable = Enumerable.Range(0, 9).OrderBy(x => Guid.NewGuid()).Take(9);
        var items = enumerable.ToArray();
        
        myArea.CleanPyramidArea();
        
        agentRb.velocity = Vector3.zero;
        myArea.PlaceObject(gameObject, items[0]);
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        switchLogic.ResetSwitch(items[1], items[2]);
        myArea.CreateStonePyramid(1, items[3]);
        myArea.CreateStonePyramid(1, items[4]);
        myArea.CreateStonePyramid(1, items[5]);
        myArea.CreateStonePyramid(1, items[6]);
        myArea.CreateStonePyramid(1, items[7]);
        myArea.CreateStonePyramid(1, items[8]);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("goal"))
        {
            SetReward(2f);
            Done();
        }
    }

    public override void AgentOnDone()
    {

    }
}
