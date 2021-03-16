using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;


public class SorterAgent : Agent
{
    [Range(1, 20)]
    public int DefaultMaxNumTiles;
    private const int k_HighestTileValue = 20;

    int m_NumberOfTilesToSpawn;
    int m_MaxNumberOfTiles;
    Rigidbody m_AgentRb;

    // The BufferSensorComponent is the Sensor that allows the Agent to observe
    // a variable number of items (here, numbered tiles)
    BufferSensorComponent m_BufferSensor;

    public List<NumberTile> NumberTilesList = new List<NumberTile>();

    private List<NumberTile> CurrentlyVisibleTilesList = new List<NumberTile>();
    private List<Transform> AlreadyTouchedList = new List<Transform>();

    private List<int> m_UsedPositionsList = new List<int>();
    private Vector3 m_StartingPos;

    GameObject m_Area;
    EnvironmentParameters m_ResetParams;

    private int m_NextExpectedTileIndex;


    public override void Initialize()
    {
        m_Area = transform.parent.gameObject;
        m_MaxNumberOfTiles = k_HighestTileValue;
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        m_BufferSensor = GetComponent<BufferSensorComponent>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_StartingPos = transform.position;
    }

    public override void OnEpisodeBegin()
    {
        m_MaxNumberOfTiles = (int)m_ResetParams.GetWithDefault("num_tiles", DefaultMaxNumTiles);

        m_NumberOfTilesToSpawn = Random.Range(1, m_MaxNumberOfTiles + 1);
        SelectTilesToShow();
        SetTilePositions();

        transform.position = m_StartingPos;
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation((transform.position.x - m_Area.transform.position.x) / 20f);
        sensor.AddObservation((transform.position.z - m_Area.transform.position.z) / 20f);

        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);

        foreach (var item in CurrentlyVisibleTilesList)
        {
            // Each observation / tile in the BufferSensor will have 22 values
            // The first 20 are one hot encoding of the value of the tile
            // The 21st and 22nd are the position of the tile relative to the agent
            // The 23rd is a boolean : 1 if the tile was visited already and 0 otherwise
            float[] listObservation = new float[k_HighestTileValue + 3];
            listObservation[item.NumberValue] = 1.0f;
            var tileTransform = item.transform.GetChild(1);
            listObservation[k_HighestTileValue] = (tileTransform.position.x - transform.position.x) / 20f;
            listObservation[k_HighestTileValue + 1] = (tileTransform.position.z - transform.position.z) / 20f;
            listObservation[k_HighestTileValue + 2] = item.IsVisited ? 1.0f : 0.0f;
            // Here, the observation for the tile is added to the BufferSensor
            m_BufferSensor.AppendObservation(listObservation);

        };

    }

    private void OnCollisionEnter(Collision col)
    {
        if (!col.gameObject.CompareTag("tile"))
        {
            return;
        }
        if (AlreadyTouchedList.Contains(col.transform))
        {
            return;
        }
        if (col.transform.parent != CurrentlyVisibleTilesList[m_NextExpectedTileIndex].transform)
        {
            // The Agent Failed
            AddReward(-1);
            EndEpisode();
        }
        else
        {
            // The Agent Succeeded
            AddReward(1);
            var tile = col.gameObject.GetComponentInParent<NumberTile>();
            tile.VisitTile();
            m_NextExpectedTileIndex++;

            AlreadyTouchedList.Add(col.transform);

            //We got all of them. Can reset now.
            if (m_NextExpectedTileIndex == m_NumberOfTilesToSpawn)
            {
                EndEpisode();
            }
        }
    }

    void SetTilePositions()
    {

        m_UsedPositionsList.Clear();

        //Disable all. We will enable the ones selected
        foreach (var item in NumberTilesList)
        {
            item.ResetTile();
            item.gameObject.SetActive(false);
        }


        foreach (var item in CurrentlyVisibleTilesList)
        {
            //Select a rnd spawnAngle
            bool posChosen = false;
            int rndPosIndx = 0;
            while (!posChosen)
            {
                rndPosIndx = Random.Range(0, k_HighestTileValue);
                if (!m_UsedPositionsList.Contains(rndPosIndx))
                {
                    m_UsedPositionsList.Add(rndPosIndx);
                    posChosen = true;
                }
            }
            item.transform.localRotation = Quaternion.Euler(0, rndPosIndx * (360f / k_HighestTileValue), 0);
            item.gameObject.SetActive(true);
        }
    }

    void SelectTilesToShow()
    {

        CurrentlyVisibleTilesList.Clear();
        AlreadyTouchedList.Clear();

        int numLeft = m_NumberOfTilesToSpawn;
        while (numLeft > 0)
        {
            int rndInt = Random.Range(0, k_HighestTileValue);
            var tmp = NumberTilesList[rndInt];
            if (!CurrentlyVisibleTilesList.Contains(tmp))
            {
                CurrentlyVisibleTilesList.Add(tmp);
                numLeft--;
            }
        }

        //Sort Ascending
        CurrentlyVisibleTilesList.Sort((x, y) => x.NumberValue.CompareTo(y.NumberValue));
        m_NextExpectedTileIndex = 0;
    }


    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var forwardAxis = act[0];
        var rightAxis = act[1];
        var rotateAxis = act[2];

        switch (forwardAxis)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
        }

        switch (rightAxis)
        {
            case 1:
                dirToGo = transform.right * 1f;
                break;
            case 2:
                dirToGo = transform.right * -1f;
                break;
        }

        switch (rotateAxis)
        {
            case 1:
                rotateDir = transform.up * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * 2, ForceMode.VelocityChange);

    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        // Move the agent using the action.
        MoveAgent(actionBuffers.DiscreteActions);

        // Penalty given each step to encourage agent to finish task quickly.
        AddReward(-1f / MaxStep);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut.Clear();
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        //rotate
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[2] = 2;
        }
        //right
        if (Input.GetKey(KeyCode.E))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            discreteActionsOut[1] = 2;
        }
    }
}
