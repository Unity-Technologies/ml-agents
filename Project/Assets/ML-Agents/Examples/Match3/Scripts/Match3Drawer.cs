using UnityEngine;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{
    public class Match3Drawer : MonoBehaviour
    {
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

        private static Color s_EmptyColor = new Color(0.5f, 0.5f, 0.5f, .25f);


        void OnDrawGizmos()
        {
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
                    var value = board.Cells != null ? board.Cells[j, i] : Match3Board.k_EmptyCell;
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

                    Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * Vector3.one);

                    Gizmos.color = Color.yellow;
                    if (board.Matched != null && board.Matched[j, i])
                    {
                        Gizmos.DrawWireCube(transform.TransformPoint(pos), matchedWireframeSize * Vector3.one);
                    }
                }
            }

            // Draw valid moves

            for (var eIdx = 0; eIdx < Move.NumPotentialMoves(board.Rows, board.Columns); eIdx++)
            {
                if (DebugEdgeIndex >= 0 && eIdx != DebugEdgeIndex)
                {
                    continue;
                }
                Move move = Move.FromMoveIndex(eIdx, board.Rows, board.Columns);
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
