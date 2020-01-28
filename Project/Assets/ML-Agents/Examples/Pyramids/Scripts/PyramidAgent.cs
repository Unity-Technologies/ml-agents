using System;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;
using MLAgents;

public class PyramidAgent : Agent
{
    public GameObject area;
    PyramidArea m_MyArea;
    Rigidbody m_AgentRb;
    PyramidSwitch m_SwitchLogic;
    public GameObject areaSwitch;
    public bool useVectorObs;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<PyramidArea>();
        m_SwitchLogic = areaSwitch.GetComponent<PyramidSwitch>();
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            AddVectorObs(m_SwitchLogic.GetState());
            AddVectorObs(transform.InverseTransformDirection(m_AgentRb.velocity));
        }
    }

    public void MoveAgent(float[] act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

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
        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * 2f, ForceMode.VelocityChange);
    }

    public override void AgentAction(float[] vectorAction)
    {
        AddReward(-1f / maxStep);
        MoveAgent(vectorAction);
    }

    public override float[] Heuristic()
    {
        if (Input.GetKey(KeyCode.D))
        {
            return new float[] { 3 };
        }
        if (Input.GetKey(KeyCode.W))
        {
            return new float[] { 1 };
        }
        if (Input.GetKey(KeyCode.A))
        {
            return new float[] { 4 };
        }
        if (Input.GetKey(KeyCode.S))
        {
            return new float[] { 2 };
        }
        return new float[] { 0 };
    }

    public override void AgentReset()
    {
        var enumerable = Enumerable.Range(0, 9).OrderBy(x => Guid.NewGuid()).Take(9);
        var items = enumerable.ToArray();

        m_MyArea.CleanPyramidArea();

        m_AgentRb.velocity = Vector3.zero;
        m_MyArea.PlaceObject(gameObject, items[0]);
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        m_SwitchLogic.ResetSwitch(items[1], items[2]);
        m_MyArea.CreateStonePyramid(1, items[3]);
        m_MyArea.CreateStonePyramid(1, items[4]);
        m_MyArea.CreateStonePyramid(1, items[5]);
        m_MyArea.CreateStonePyramid(1, items[6]);
        m_MyArea.CreateStonePyramid(1, items[7]);
        m_MyArea.CreateStonePyramid(1, items[8]);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("goal"))
        {
            SetReward(2f);
            Done();
        }
    }
}
