using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

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

        public Dictionary<(int, int), Match3TileSelector> tilesDict = new Dictionary<(int, int), Match3TileSelector>();
        public float CubeSpacing = 1.25f;
        public GameObject TilePrefab;

        private bool m_Initialized;
        private Match3Board m_Board;

        void Awake()
        {
            if (!m_Initialized)
            {
                InitializeDict();
            }
        }

        void InitializeDict()
        {
            m_Board = GetComponent<Match3Board>();
            foreach (var item in tilesDict)
            {
                if (item.Value)
                {
                    DestroyImmediate(item.Value.gameObject);
                }
            }

            tilesDict.Clear();

            for (var i = 0; i < m_Board.MaxRows; i++)
            {
                for (var j = 0; j < m_Board.MaxColumns; j++)
                {
                    var go = Instantiate(TilePrefab, transform.position, Quaternion.identity, transform);
                    go.name = $"r{i}_c{j}";
                    tilesDict.Add((i, j), go.GetComponent<Match3TileSelector>());
                }
            }

            m_Initialized = true;
        }

        void Update()
        {
            if (!m_Board)
            {
                m_Board = GetComponent<Match3Board>();
            }

            if (!m_Initialized)
            {
                InitializeDict();
            }

            var currentSize = m_Board.GetCurrentBoardSize();
            for (var i = 0; i < m_Board.MaxRows; i++)
            {
                for (var j = 0; j < m_Board.MaxColumns; j++)
                {
                    int value = Match3Board.k_EmptyCell;
                    int specialType = 0;
                    if (m_Board.Cells != null && i < currentSize.Rows && j < currentSize.Columns)
                    {
                        value = m_Board.GetCellType(i, j);
                        specialType = m_Board.GetSpecialType(i, j);
                    }
                    var pos = new Vector3(j, i, 0);
                    pos *= CubeSpacing;

                    tilesDict[(i, j)].transform.position = transform.TransformPoint(pos);
                    tilesDict[(i, j)].SetActiveTile(specialType, value);
                }
            }
        }

        void OnDrawGizmos()
        {
            Profiler.BeginSample("Match3.OnDrawGizmos");
            var cubeSize = .5f;
            var matchedWireframeSize = .5f * (cubeSize + CubeSpacing);

            if (!m_Board)
            {
                m_Board = GetComponent<Match3Board>();
                if (m_Board == null)
                {
                    return;
                }
            }

            var currentSize = m_Board.GetCurrentBoardSize();
            for (var i = 0; i < m_Board.MaxRows; i++)
            {
                for (var j = 0; j < m_Board.MaxColumns; j++)
                {
                    int value = Match3Board.k_EmptyCell;
                    int specialType = 0;
                    if (m_Board.Cells != null && i < currentSize.Rows && j < currentSize.Columns)
                    {
                        value = m_Board.GetCellType(i, j);
                        specialType = m_Board.GetSpecialType(i, j);
                    }

                    if (value >= 0 && value < s_Colors.Length)
                    {
                        Gizmos.color = s_Colors[value];
                    }
                    else
                    {
                        Gizmos.color = s_EmptyColor;
                    }

                    var pos = new Vector3(j, i, 0);
                    pos *= CubeSpacing;

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
                    if (m_Board.Matched != null && m_Board.Matched[j, i])
                    {
                        Gizmos.DrawWireCube(transform.TransformPoint(pos), matchedWireframeSize * Vector3.one);
                    }
                }
            }

            // Draw valid moves
            foreach (var move in m_Board.AllMoves())
            {
                if (DebugMoveIndex >= 0 && move.MoveIndex != DebugMoveIndex)
                {
                    continue;
                }

                if (!m_Board.IsMoveValid(move))
                {
                    continue;
                }

                var (otherRow, otherCol) = move.OtherCell();
                var pos = new Vector3(move.Column, move.Row, 0) * CubeSpacing;
                var otherPos = new Vector3(otherCol, otherRow, 0) * CubeSpacing;

                var oneQuarter = Vector3.Lerp(pos, otherPos, .25f);
                var threeQuarters = Vector3.Lerp(pos, otherPos, .75f);
                Gizmos.DrawLine(transform.TransformPoint(oneQuarter), transform.TransformPoint(threeQuarters));
            }

            Profiler.EndSample();
        }
    }
}
