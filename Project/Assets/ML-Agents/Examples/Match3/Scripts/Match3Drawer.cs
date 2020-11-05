using System;
using System.Collections.Generic;
using Unity.Mathematics;
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


        public Dictionary<(int, int), Match3TileSelector> tilesDict = new Dictionary<(int, int), Match3TileSelector>();
        public bool initialized;
        public bool clearDict;
        public float cubeSpacing = 1;
        public Match3Board board;
        void InitializeDict()
        {
            board = GetComponent<Match3Board>();
            foreach (var item in tilesDict)
            {
                if (item.Value)
                {
                    DestroyImmediate(item.Value.gameObject);
                }
            }
            tilesDict.Clear();

            //            if (!initialized)
            //            {
            for (var i = 0; i < board.Rows; i++)
            {
                for (var j = 0; j < board.Columns; j++)
                {
                    var go = Instantiate(board.tilePrefab, transform.position, quaternion.identity, transform);
                    go.name = $"r{i}_c{j}";
                    tilesDict.Add((i, j), go.GetComponent<Match3TileSelector>());
                    //                    tilesDict[(i, j)].SetActiveTile(0, 0);
                    //                        tilesDict[item.Key] = go.GetComponent<Match3TileSelector>();
                    //                        tilesDict[item.Key].SetActiveTile(0);


                    //                        tilesDict.Add((i, j), null);
                }
            }
            initialized = true;
            //            }
        }

        void Update()
        {
            if (!board)
            {
                board = GetComponent<Match3Board>();
            }
            if (!initialized)
            {
                InitializeDict();
            }

            //            var cubeSize = .5f;

            for (var i = 0; i < board.Rows; i++)
            {
                for (var j = 0; j < board.Columns; j++)
                {
                    var value = board.Cells != null ? board.GetCellType(i, j) : Match3Board.k_EmptyCell;
                    //                    var go = board.blankTile;
                    //                    if (value >= 0 && value < s_Colors.Length)
                    //                    {
                    //                        go = board.tilePrefabs[0];
                    ////                        Gizmos.color = s_Colors[value];
                    //                    }
                    //                    else
                    //                    {
                    //                        Gizmos.color = s_EmptyColor;
                    //                    }

                    var pos = new Vector3(j, i, 0);
                    pos *= cubeSpacing;

                    var specialType = board.Cells != null ? board.GetSpecialType(i, j) : 0;
                    //                    if (specialType == 2)
                    //                    {
                    ////                        tilesDict[(i, j)].SetActiveTile(8);
                    //                        tilesDict[(i, j)].SetActiveTile(2, value + 1);
                    //                    }
                    //                    else if (specialType == 1)
                    //                    {
                    //                        tilesDict[(i, j)].SetActiveTile(1, value + 1);
                    //                    }
                    //                    else
                    //                    {
                    //                        tilesDict[(i, j)].SetActiveTile(value + 1);
                    //                        //                        print(value + 1);
                    //                    }
                    tilesDict[(i, j)].transform.position = transform.TransformPoint(pos);
                    tilesDict[(i, j)].SetActiveTile(specialType, value + 1);

                }
            }

        }

        void OnDrawGizmos()
        {
            // TODO replace Gizmos for drawing the game state with proper GameObjects and animations.
            var cubeSize = .5f;
            //            var cubeSpacing = .75f;
            var matchedWireframeSize = .5f * (cubeSize + cubeSpacing);

            //            var board = GetComponent<Match3Board>();
            //            if (board == null)
            //            {
            //                return;
            //            }
            if (!board)
            {
                board = GetComponent<Match3Board>();
            }

            //            if (clearDict)
            //            {
            //                 for (var i = 0; i < board.Rows; i++)
            //                 {
            //                       for (var j = 0; j < board.Columns; j++)
            //                       {
            //                           if (tilesDict.ContainsKey((i, j)))
            //                           {
            //                               DestroyImmediate(tilesDict[(i, j)].gameObject);
            //                           }
            //                       }
            //                 }
            //                tilesDict.Clear();
            //                clearDict = false;
            //            }

            //            if (!initialized)
            //            {
            //                InitializeDict();
            ////                 for (var i = 0; i < board.Rows; i++)
            ////                 {
            ////                       for (var j = 0; j < board.Columns; j++)
            ////                       {
            ////                           var go =  Instantiate(board.tilePrefabs[0], transform.position, quaternion.identity, transform);
            ////
            ////                            tilesDict[(i, j)] = go.GetComponent<Match3TileSelector>();
            ////                            tilesDict[(i, j)].SetActiveTile(0);
            ////                       }
            ////                 }
            //            }
            //                foreach (var item in tilesDict)
            //                {
            ////                    var go = item.Value;
            //                    if (!item.Value)
            //                    {
            //                        var go =  Instantiate(board.tilePrefabs[0], transform.position, quaternion.identity, transform);
            //                        tilesDict[item.Key] = go.GetComponent<Match3TileSelector>();
            //                        tilesDict[item.Key].SetActiveTile(0);
            //                    }
            //
            //                }

            for (var i = 0; i < board.Rows; i++)
            {
                for (var j = 0; j < board.Columns; j++)
                {
                    var value = board.Cells != null ? board.GetCellType(i, j) : Match3Board.k_EmptyCell;
                    //                    var go = board.blankTile;
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
                        //                        tilesDict[(i, j)].SetActiveTile(7);
                    }
                    else if (specialType == 1)
                    {
                        Gizmos.DrawSphere(transform.TransformPoint(pos), .5f * cubeSize);
                        //                        tilesDict[(i,` j)].SetActiveTile(6);
                    }
                    else
                    {
                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * Vector3.one);
                        //                        tilesDict[(i, j)].transform.position = transform.TransformPoint(pos);
                        //                        tilesDict[(i, j)].SetActiveTile(value);
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

        //        void OnDrawGizmos()
        //        {
        //            // TODO replace Gizmos for drawing the game state with proper GameObjects and animations.
        //            var cubeSize = .5f;
        //            var cubeSpacing = .75f;
        //            var matchedWireframeSize = .5f * (cubeSize + cubeSpacing);
        //
        //            var board = GetComponent<Match3Board>();
        //            if (board == null)
        //            {
        //                return;
        //            }
        //
        //            for (var i = 0; i < board.Rows; i++)
        //            {
        //                for (var j = 0; j < board.Columns; j++)
        //                {
        //                    var value = board.Cells != null ? board.GetCellType(i, j) : Match3Board.k_EmptyCell;
        //                    if (value >= 0 && value < s_Colors.Length)
        //                    {
        //                        Gizmos.color = s_Colors[value];
        //                    }
        //                    else
        //                    {
        //                        Gizmos.color = s_EmptyColor;
        //                    }
        //
        //                    var pos = new Vector3(j, i, 0);
        //                    pos *= cubeSpacing;
        //
        //                    var specialType = board.Cells != null ? board.GetSpecialType(i, j) : 0;
        //                    if (specialType == 2)
        //                    {
        //                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(1f, .5f, .5f));
        //                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(.5f, 1f, .5f));
        //                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * new Vector3(.5f, .5f, 1f));
        //                    }
        //                    else if (specialType == 1)
        //                    {
        //                        Gizmos.DrawSphere(transform.TransformPoint(pos), .5f * cubeSize);
        //                    }
        //                    else
        //                    {
        //                        Gizmos.DrawCube(transform.TransformPoint(pos), cubeSize * Vector3.one);
        //                    }
        //
        //                    Gizmos.color = Color.yellow;
        //                    if (board.Matched != null && board.Matched[j, i])
        //                    {
        //                        Gizmos.DrawWireCube(transform.TransformPoint(pos), matchedWireframeSize * Vector3.one);
        //                    }
        //                }
        //            }
        //
        //            // Draw valid moves
        //            foreach (var move in board.AllMoves())
        //            {
        //                if (DebugMoveIndex >= 0 && move.MoveIndex != DebugMoveIndex)
        //                {
        //                    continue;
        //                }
        //
        //                if (!board.IsMoveValid(move))
        //                {
        //                    continue;
        //                }
        //
        //                var (otherRow, otherCol) = move.OtherCell();
        //                var pos = new Vector3(move.Column, move.Row, 0) * cubeSpacing;
        //                var otherPos = new Vector3(otherCol, otherRow, 0) * cubeSpacing;
        //
        //                var oneQuarter = Vector3.Lerp(pos, otherPos, .25f);
        //                var threeQuarters = Vector3.Lerp(pos, otherPos, .75f);
        //                Gizmos.DrawLine(transform.TransformPoint(oneQuarter), transform.TransformPoint(threeQuarters));
        //            }
        //        }


    }
}
