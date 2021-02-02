using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;


public class SequencerAgent : Agent
{

    // class TileInfo
    // {
    //     public SequenceTile tile;
    //     public int SpawnIndexPos;
    // }
    public bool SelectNewTiles;

    public int NumberOfTilesToSpawn;
    PushBlockSettings m_PushBlockSettings;
    Rigidbody m_AgentRb;  //cached on initialization

    public List<SequenceTile> SequenceTilesList = new List<SequenceTile>();
    public List<SequenceTile> CurrentlyVisibleTilesList = new List<SequenceTile>();
    private List<Transform> AlreadyTouchedList = new List<Transform>();

    public List<int> m_UsedPositionsList = new List<int>();
    private Vector3 m_StartingPos;



    // private SequenceTile m_NextExpectedTile;
    public int m_NextExpectedTileIndex;
    public Material TileMaterial;
    public Material SuccessMaterial;
    public override void Initialize()
    {
        m_PushBlockSettings = FindObjectOfType<PushBlockSettings>();
        m_AgentRb = GetComponent<Rigidbody>();
        m_StartingPos = transform.position;
    }


    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {

        SelectTilesToShow();
        SetTilePositions();

        transform.position = m_StartingPos;
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;
    }


    private void Update()
    {
        //DEBUG
        if (SelectNewTiles)
        {
            SelectNewTiles = false;
            SelectTilesToShow();
            SetTilePositions();
        }
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // foreach (var item in CurrentlyVisibleTilesList)
        // {
        //     sensor.AddObservation(item.transform.localRotation.y / 360);
        // }
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
            //failed
            AddReward(-1);
            EndEpisode();
            print("no");
        }
        else
        {
            //success
            print("yes");
            AddReward(1);
            var tile = col.gameObject.GetComponentInParent<SequenceTile>();
            tile.rend.sharedMaterial = SuccessMaterial;
            m_NextExpectedTileIndex++;

            AlreadyTouchedList.Add(col.transform);
            //We got all of them. Can reset now.
            if (m_NextExpectedTileIndex == NumberOfTilesToSpawn)
            {
                EndEpisode();
            }
        }
    }

    void SetTilePositions()
    {

        m_UsedPositionsList.Clear();

        //Disable all. We will enable the ones selected
        foreach (var item in SequenceTilesList)
        {
            item.gameObject.SetActive(false);
        }


        foreach (var item in CurrentlyVisibleTilesList)
        {
            //Select a rnd spawnAngle
            bool posChosen = false;
            int rndPosIndx = 0;
            while (!posChosen)
            {
                rndPosIndx = Random.Range(0, SequenceTilesList.Count);
                if (!m_UsedPositionsList.Contains(rndPosIndx))
                {
                    m_UsedPositionsList.Add(rndPosIndx);
                    posChosen = true;
                }
            }
            item.transform.localRotation = Quaternion.Euler(0, rndPosIndx * (360f / SequenceTilesList.Count), 0);
            item.rend.sharedMaterial = TileMaterial;
            item.gameObject.SetActive(true);
        }
    }

    void SelectTilesToShow()
    {
        // Shuffle(SequenceTilesList);
        // Random m_RandomTile = new Random();

        CurrentlyVisibleTilesList.Clear();
        AlreadyTouchedList.Clear();

        int numLeft = NumberOfTilesToSpawn;
        while (numLeft > 0)
        {
            int rndInt = Random.Range(0, SequenceTilesList.Count);
            var tmp = SequenceTilesList[rndInt];
            if (!CurrentlyVisibleTilesList.Contains(tmp))
            {
                CurrentlyVisibleTilesList.Add(tmp);
                numLeft--;
            }
        }

        //Sort Ascending
        CurrentlyVisibleTilesList.Sort((x, y) => x.NumberValue.CompareTo(y.NumberValue));
        // m_NextExpectedTile = CurrentlyVisibleTilesList[0];
        m_NextExpectedTileIndex = 0;
    }


    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

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
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
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
        discreteActionsOut[0] = 0;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }
}
