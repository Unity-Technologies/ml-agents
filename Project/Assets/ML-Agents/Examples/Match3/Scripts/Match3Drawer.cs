using UnityEngine;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3Drawer : MonoBehaviour
    {
        public int DebugMoveIndex = -1;

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

        private static Color s_EmptyColor = new Color(0.5f, 0.5f, 0.5f, .25f);


        void OnDrawGizmos()
        {
            // TODO replace Gizmos for drawing the game state with proper GameObjects and animations.
            var cubeSize = .5f;
            var cubeSpacing = .75f;
            var matchedWireframeSize = .5f * (cubeSize + cubeSpacing);

            var board = GetComponent<Match3Board>();
            if (board == null)
            {
                return;
            }

            for (var i = 0; i < board.Rows; i++)
            {
                for (var j = 0; j < board.Columns; j++)
                {
                    var value = board.Cells != null ? board.GetCellType(i, j) : Match3Board.k_EmptyCell;
                    if (value >= 0 && value < s_Colors.Length)
                    {
                        Gizmos.color = s_Colors[value];
                    }
                    else
                    {
                        Gizmos.color = s_EmptyColor;
                    }

                    var pos = new Vector3(j, i, 0);
                    pos *= cubeSpacing;

                    var specialType = board.Cells != null ? board.GetSpecialType(i, j) : 0;
                    if (specialType == 2)
                    {
                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(1f, .5f, .5f));
                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(.5f, 1f, .5f));
                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(.5f, .5f, 1f));
                    }
                    else if (specialType == 1)
                    {
                        Gizmos.DrawSphere(transform.TransformPoint(pos), .5f * cubeSize);
                    }
                    else
                    {
                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * Vector3.one);
                    }

                    Gizmos.color = Color.yellow;
                    if (board.Matched != null && board.Matched[j, i])
                    {
                        Gizmos.DrawWireCube(transform.TransformPoint(pos), matchedWireframeSize * Vector3.one);
                    }
                }
            }

            // Draw valid moves
            foreach (var move in board.AllMoves())
            {
                if (DebugMoveIndex >= 0 && move.MoveIndex != DebugMoveIndex)
                {
                    continue;
                }

                if (!board.IsMoveValid(move))
                {
                    continue;
                }

                var (otherRow, otherCol) = move.OtherCell();
                var pos = new Vector3(move.Column, move.Row, 0) * cubeSpacing;
                var otherPos = new Vector3(otherCol, otherRow, 0) * cubeSpacing;

                var oneQuarter = Vector3.Lerp(pos, otherPos, .25f);
                var threeQuarters = Vector3.Lerp(pos, otherPos, .75f);
                Gizmos.DrawLine(transform.TransformPoint(oneQuarter), transform.TransformPoint(threeQuarters));
            }
        }
    }
}
