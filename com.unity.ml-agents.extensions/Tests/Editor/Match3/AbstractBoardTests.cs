using System;
using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgents.Extensions.Tests.Match3
{
    internal class StringBoard : AbstractBoard
    {
        private string[] m_Board;

        /// <summary>
        /// Convert a string like "000\n010\n000" to a board representation
        /// Row 0 is considered the bottom row
        /// </summary>
        /// <param name="newBoard"></param>
        public void SetBoard(string newBoard)
        {
            m_Board = newBoard.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            Rows = m_Board.Length;
            Columns = m_Board[0].Length;
            NumCellTypes = 0;
            for (var r = 0; r < Rows; r++)
            {
                for (var c = 0; c < Columns; c++)
                {
                    NumCellTypes = Mathf.Max(NumCellTypes, 1 + GetCellType(r, c));
                }
            }
        }

        public override bool MakeMove(Move m)
        {
            return true;
        }

        public override bool IsMoveValid(Move m)
        {
            return SimpleIsMoveValid(m);
        }

        public override int GetCellType(int row, int col)
        {
            var character = m_Board[m_Board.Length - 1 - row][col];
            return (int)(character - '0');
        }

        public override int GetSpecialType(int row, int col)
        {
            return 0;
        }

    }

    public class AbstractBoardTests
    {
        [Test]
        public void TestBoardInit()
        {
            var boardString =
                @"000
                  000
                  010";
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();
            board.SetBoard(boardString);

            Assert.AreEqual(3, board.Rows);
            Assert.AreEqual(3, board.Columns);
            Assert.AreEqual(2, board.NumCellTypes);
            for (var r = 0; r < 3; r++)
            {
                for (var c = 0; c < 3; c++)
                {
                    var expected = (r == 0 && c == 1) ? 1 : 0;
                    Assert.AreEqual(expected, board.GetCellType(r, c));
                }
            }
        }

        [Test]
        public void TestCheckValidMoves()
        {
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();

            var boardString =
                @"0105
                  1024
                  0203
                  2022";
            board.SetBoard(boardString);

            var validMoves = new[]
            {
                Move.FromPositionAndDirection(2, 1, Direction.Up, board.Rows, board.Columns), // equivalent to (3, 1, Down)
                Move.FromPositionAndDirection(2, 1, Direction.Left, board.Rows, board.Columns), // equivalent to (2, 0, Right)
                Move.FromPositionAndDirection(2, 1, Direction.Down, board.Rows, board.Columns), // equivalent to (1, 1, Up)
                Move.FromPositionAndDirection(2, 1, Direction.Right, board.Rows, board.Columns),
                Move.FromPositionAndDirection(1, 1, Direction.Down, board.Rows, board.Columns),
                Move.FromPositionAndDirection(1, 1, Direction.Left, board.Rows, board.Columns),
                Move.FromPositionAndDirection(1, 1, Direction.Right, board.Rows, board.Columns),
                Move.FromPositionAndDirection(0, 1, Direction.Left, board.Rows, board.Columns),
            };

            foreach (var m in validMoves)
            {
                Assert.IsTrue(board.IsMoveValid(m));
            }

            // Run through all moves and make sure those are the only valid ones
            HashSet<int> validIndices = new HashSet<int>();
            foreach (var m in validMoves)
            {
                validIndices.Add(m.MoveIndex);
            }

            var numPotentialMoves = Move.NumPotentialMoves(board.Rows, board.Columns);
            for (var i = 0; i < numPotentialMoves; i++)
            {
                var move = Move.FromMoveIndex(i, board.Rows, board.Columns);
                var expected = validIndices.Contains(move.MoveIndex);
                Assert.AreEqual(expected, board.IsMoveValid(move), $"({move.Row}, {move.Column}, {move.Direction})");
            }
        }

    }
}
