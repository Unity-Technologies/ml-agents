using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public enum BoardStatus
{
    Empty,
    FilledX,
    FilledO
}

public enum WinState
{
    NoWinnerYet,
    WinnerX,
    WinnerO,
    Draw
}

public class TicTacToeGame : MonoBehaviour
{
    public BoardStatus[,] Board = new BoardStatus[3, 3];
    private PlayerType m_CurrentPlayer = PlayerType.PlayerX;
    private TicTacToeAgent m_AgentX;
    private TicTacToeAgent m_AgentO;

    // Start is called before the first frame update
    void Start()
    {
        Academy.Instance.AutomaticSteppingEnabled = false;
        var agents = GetComponentsInChildren<TicTacToeAgent>();
        foreach (var agent in agents)
        {
            if (agent.playerType == PlayerType.PlayerX)
            {
                m_AgentX = agent;
            }
            else if (agent.playerType == PlayerType.PlayerO)
            {
                m_AgentO = agent;
            }
            else
            {
                throw new UnityAgentsException("Unknown player type.");
            }
        }

        Debug.Assert(m_AgentX != null);
        Debug.Assert(m_AgentO != null);
    }

    void InitBoard()
    {
        var rows = Board.GetLength(0);
        var cols = Board.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                Board[r, c] = BoardStatus.Empty;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        Agent currentAgent = (m_CurrentPlayer == PlayerType.PlayerX) ? m_AgentX : m_AgentO;
        currentAgent.RequestDecision();
        Academy.Instance.EnvironmentStep();

        var winState = CheckForWin();
        if (winState == WinState.NoWinnerYet)
        {
            // Swap
            m_CurrentPlayer = m_CurrentPlayer == PlayerType.PlayerX ? PlayerType.PlayerO : PlayerType.PlayerX;
        }
        else
        {
            //Debug.Log(this.ToString());
            float rewardX = 0.0f;
            float rewardO = 0.0f;

            if (winState == WinState.WinnerX)
            {
                rewardX = 1.0f;
                rewardO = -1.0f;
                //Debug.Log("X wins!");
            }
            else if (winState == WinState.WinnerO)
            {
                rewardX = -1.0f;
                rewardO = 1.0f;
                //Debug.Log("O wins!");
            }
            else
            {
                //Debug.Log("Nobody wins :(");
            }
            m_AgentX.AddReward(rewardX);
            m_AgentX.EndEpisode();

            m_AgentO.AddReward(rewardO);
            m_AgentO.EndEpisode();

            InitBoard();
        }
    }

    public WinState CheckForWin()
    {
        // Assume 3 rows and columns for now, not interested in generalizing.

        // Check rows
        for (var r = 0; r < 3; r++)
        {
            if (Board[r, 0] == BoardStatus.Empty)
            {
                continue;
            }

            if (Board[r, 1] == Board[r, 0] && Board[r, 2] == Board[r, 0])
            {
                // all the same in the row
                return (Board[r, 0] == BoardStatus.FilledX) ? WinState.WinnerX : WinState.WinnerO;
            }
        }

        for (var c = 0; c < 3; c++)
        {
            if (Board[0, c] == BoardStatus.Empty)
            {
                continue;
            }

            if (Board[1, c] == Board[0, c] && Board[2, c] == Board[0, c])
            {
                // all the same in the column
                return (Board[0, c] == BoardStatus.FilledX) ? WinState.WinnerX : WinState.WinnerO;
            }
        }

        // Check diagonals
        {
            if (Board[0, 0] == Board[1, 1] && Board[0, 0] == Board[2, 2] && Board[0, 0] != BoardStatus.Empty)
            {
                return (Board[0, 0] == BoardStatus.FilledX) ? WinState.WinnerX : WinState.WinnerO;
            }

            if (Board[0, 2] == Board[1, 1] && Board[0, 2] == Board[2, 0] && Board[0, 2] != BoardStatus.Empty)
            {
                return (Board[0, 2] == BoardStatus.FilledX) ? WinState.WinnerX : WinState.WinnerO;
            }
        }

        // No winner, check if there are still moves or it's a draw
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 3; c++)
            {
                if (Board[r, c] == BoardStatus.Empty)
                {
                    return WinState.NoWinnerYet;
                }
            }
        }

        // No winner, no empty squares, do it's a draw
        return WinState.Draw;
    }

    static string BoardStatusToString(BoardStatus s)
    {
        switch (s)
        {
            case BoardStatus.Empty:
                return ".";
            case BoardStatus.FilledX:
                return "X";
            case BoardStatus.FilledO:
                return "O";
            default:
                throw new ArgumentOutOfRangeException(nameof(s), s, null);
        }
    }

    public override string ToString()
    {
        var b00 = BoardStatusToString(Board[0, 0]);
        var b01 = BoardStatusToString(Board[0, 1]);
        var b02 = BoardStatusToString(Board[0, 2]);

        var b10 = BoardStatusToString(Board[1, 0]);
        var b11 = BoardStatusToString(Board[1, 1]);
        var b12 = BoardStatusToString(Board[1, 2]);

        var b20 = BoardStatusToString(Board[2, 0]);
        var b21 = BoardStatusToString(Board[2, 1]);
        var b22 = BoardStatusToString(Board[2, 2]);
        return $"\n{b00}{b01}{b02}\n{b10}{b11}{b12}\n{b20}{b21}{b22}";
    }
}
