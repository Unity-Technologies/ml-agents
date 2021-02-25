using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public enum PlayerType
{
    PlayerX,
    PlayerO
}

public class TicTacToeAgent : Agent
{
    public PlayerType playerType;

    private TicTacToeGame m_Game;
    private System.Random m_Random;

    // Start is called before the first frame update
    void Start()
    {
        m_Game = GetComponentInParent<TicTacToeGame>();
        m_Random = new System.Random(GetInstanceID());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var rows = m_Game.Board.GetLength(0);
        var cols = m_Game.Board.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                // One-hot encoding of the board state *depending on the agent's player type*
                // Set the 0th element if the board is empty
                // Set the 1st element if the board is my type
                // Set the 2nd element if the board is my opponent's type
                BoardStatus boardVal = m_Game.Board[r, c];
                bool isMine = (boardVal == BoardStatus.FilledX && playerType == PlayerType.PlayerX) ||
                              (boardVal == BoardStatus.FilledO && playerType == PlayerType.PlayerO);
                int oneHotIndex = (boardVal == BoardStatus.Empty) ? 0 : (isMine) ? 1 : 2;
                sensor.AddOneHotObservation(oneHotIndex, 3);
            }
        }
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        var rows = m_Game.Board.GetLength(0);
        var cols = m_Game.Board.GetLength(1);
        int i = 0;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                BoardStatus boardVal = m_Game.Board[r, c];
                // Disallow moves on non-empty squares
                if (boardVal != BoardStatus.Empty)
                {
                    actionMask.WriteMask(0, new[] { i });
                }
                i++;
            }
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var rows = m_Game.Board.GetLength(0);
        var index = actions.DiscreteActions[0];
        var r = index / rows;
        var c = index % rows;

        // The trainer can occasionally (about 1 in a million steps) select an invalid move.
        // So make sure to check the square is empty before setting it.
        if (m_Game.Board[r, c] == BoardStatus.Empty)
        {
            m_Game.Board[r, c] = (playerType == PlayerType.PlayerX)
                ? BoardStatus.FilledX
                : BoardStatus.FilledO;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Pick a random empty square.
        var validMoves = new List<int>();
        var rows = m_Game.Board.GetLength(0);
        var cols = m_Game.Board.GetLength(1);
        int i = 0;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (m_Game.Board[r, c] == BoardStatus.Empty)
                {
                    validMoves.Add(i);
                }

                i++;
            }
        }

        var choice = validMoves[m_Random.Next(validMoves.Count)];
        var discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = choice;
    }
}
