using System;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class Match3Drawer : MonoBehaviour
    {
        public Match3Agent Agent;

        static Color[] s_Colors = new[]
        {
          Color.red,
          Color.green,
          Color.blue,
          Color.cyan,
          Color.magenta,
          Color.yellow
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

                    // Draw possible moves
                    var arrowSize = .375f;
                    if (board.IsMoveValid(i, j, Direction.Up))
                    {
                        Gizmos.DrawRay(pos, arrowSize * Vector3.up);
                    }

                    if (board.IsMoveValid(i, j, Direction.Down))
                    {
                        Gizmos.DrawRay(pos, arrowSize * Vector3.down);
                    }

                    if (board.IsMoveValid(i, j, Direction.Left))
                    {
                        Gizmos.DrawRay(pos, arrowSize * Vector3.left);
                    }

                    if (board.IsMoveValid(i, j, Direction.Right))
                    {
                        Gizmos.DrawRay(pos, arrowSize * Vector3.right);
                    }
                }
            }

        }
    }
}
