using System;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class Match3Drawer : MonoBehaviour
    {
        public Match3Agent Agent;
        public int DebugEdgeIndex = -1;

        static Color[] s_Colors = new[]
        {
          Color.red,
          Color.green,
          Color.blue,
          Color.cyan,
          Color.magenta,
          Color.yellow,
          Color.gray,
          Color.black,
        };


        void OnDrawGizmos()
        {
            var board = Agent?.Board;
            if (board == null)
            {
                return;

            }

            for (var i = 0; i < board.Rows; i++)
            {
                for (var j = 0; j < board.Columns; j++)
                {
                    var value = board.Cells[j, i];
                    if (value >= 0 && value < s_Colors.Length)
                    {
                        Gizmos.color = s_Colors[value];
                    }
                    else
                    {
                        Gizmos.color = Color.clear;
                    }

                    var pos = new Vector3(j, i, 0);

                    Gizmos.DrawCube(pos, .5f * Vector3.one);

                    Gizmos.color = Color.yellow;
                    if (board.Matched[j, i])
                    {
                        Gizmos.DrawWireCube(pos, .75f * Vector3.one);
                    }
                }
            }

            // Draw valid moves

            for (var eIdx = 0; eIdx < Move.NumEdgeIndices(board.Rows, board.Columns); eIdx++)
            {
                if (DebugEdgeIndex >= 0 && eIdx != DebugEdgeIndex)
                {
                    continue;
                }
                Move move = Move.FromEdgeIndex(eIdx, board.Rows, board.Columns);
                if (!board.IsMoveValid(move))
                {
                    continue;
                }
                var (otherRow, otherCol) = move.OtherCell();
                var pos = new Vector3(move.m_Column, move.m_Row, 0);
                var otherPos = new Vector3(otherCol, otherRow, 0);

                var oneQuarter = Vector3.Lerp(pos, otherPos, .25f);
                var threeQuarters = Vector3.Lerp(pos, otherPos, .75f);
                Gizmos.DrawLine(oneQuarter, threeQuarters);
            }
        }
    }
}
