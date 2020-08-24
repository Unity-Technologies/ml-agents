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
                    if (board.Matched[j, i])
                    {
                        Gizmos.color = Color.yellow;
                        Gizmos.DrawWireCube(pos, .75f * Vector3.one);
                    }
                }
            }

        }
    }
}
