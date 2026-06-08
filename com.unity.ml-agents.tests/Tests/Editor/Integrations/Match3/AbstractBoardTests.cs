using System;
using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using Unity.MLAgents.Integrations.Match3;

namespace Unity.MLAgents.Tests.Integrations.Match3
{
    internal class StringBoard : AbstractBoard
    {
        internal int MaxRows;
        internal int MaxColumns;
        internal int NumCellTypes;
        internal int NumSpecialTypes;
        public int CurrentRows;
        public int CurrentColumns;

        public override BoardSize GetMaxBoardSize()
        {
            return new BoardSize
            {
                Rows = MaxRows,
                Columns = MaxColumns,
                NumCellTypes = NumCellTypes,
                NumSpecialTypes = NumSpecialTypes
            };
        }

        public override BoardSize GetCurrentBoardSize()
        {
            return new BoardSize
            {
                Rows = CurrentRows,
                Columns = CurrentColumns,
                NumCellTypes = NumCellTypes,
                NumSpecialTypes = NumSpecialTypes
            };
        }

        private string[] m_Board;
        private string[] m_Special;

        /// <summary>
        /// Convert a string like "000\n010\n000" to a board representation
        /// Row 0 is considered the bottom row
        /// </summary>
        /// <param name="newBoard"></param>
        public void SetBoard(string newBoard)
        {
            m_Board = newBoard.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            MaxRows = m_Board.Length;
            MaxColumns = m_Board[0].Length;
            CurrentRows = MaxRows;
            CurrentColumns = MaxColumns;
            NumCellTypes = 0;
            for (var r = 0; r < MaxRows; r++)
            {
                for (var c = 0; c < MaxColumns; c++)
                {
                    NumCellTypes = Mathf.Max(NumCellTypes, 1 + GetCellType(r, c));
                }
            }
        }

        public void SetSpecial(string newSpecial)
        {
            m_Special = newSpecial.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            Debug.Assert(MaxRows == m_Special.Length);
            Debug.Assert(MaxColumns == m_Special[0].Length);
            NumSpecialTypes = 0;
            for (var r = 0; r < MaxRows; r++)
            {
                for (var c = 0; c < MaxColumns; c++)
                {
                    NumSpecialTypes = Mathf.Max(NumSpecialTypes, GetSpecialType(r, c));
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
            if (row >= CurrentRows || col >= CurrentColumns)
            {
                throw new IndexOutOfRangeException("Tried to get celltype out of bounds");
            }

            var character = m_Board[m_Board.Length - 1 - row][col];
            return (character - '0');
        }

        public override int GetSpecialType(int row, int col)
        {
            if (row >= CurrentRows || col >= CurrentColumns)
            {
                throw new IndexOutOfRangeException("Tried to get specialtype out of bounds");
            }

            var character = m_Special[m_Board.Length - 1 - row][col];
            return (character - '0');
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

            var boardSize = board.GetMaxBoardSize();

            Assert.AreEqual(3, boardSize.Rows);
            Assert.AreEqual(3, boardSize.Columns);
            Assert.AreEqual(2, boardSize.NumCellTypes);
            for (var r = 0; r < 3; r++)
            {
                for (var c = 0; c < 3; c++)
                {
                    var expected = (r == 0 && c == 1) ? 1 : 0;
                    Assert.AreEqual(expected, board.GetCellType(r, c));
                }
            }
        }

        internal static List<Move> GetValidMoves4x4(bool fullBoard, BoardSize boardSize)
        {
            var validMoves = new List<Move>
            {
                Move.FromPositionAndDirection(2, 1, Direction.Down, boardSize), // equivalent to (1, 1, Up)
                Move.FromPositionAndDirection(1, 1, Direction.Down, boardSize),
                Move.FromPositionAndDirection(1, 1, Direction.Left, boardSize),
                Move.FromPositionAndDirection(1, 1, Direction.Right, boardSize),
                Move.FromPositionAndDirection(0, 1, Direction.Left, boardSize),
            };

            if (fullBoard)
            {
                // This would move out of range on the small board
                // Equivalent to (3, 1, Down)
                validMoves.Add(Move.FromPositionAndDirection(2, 1, Direction.Up, boardSize));

                // These moves require matching with a cell that's off the small board, so they're invalid
                // (even though the move itself doesn't go out of range).
                validMoves.Add(Move.FromPositionAndDirection(2, 1, Direction.Left, boardSize)); // Equivalent to (2, 0, Right)
                validMoves.Add(Move.FromPositionAndDirection(2, 1, Direction.Right, boardSize));
            }

            return validMoves;
        }

        [TestCase(true, TestName = "Full Board")]
        [TestCase(false, TestName = "Small Board")]
        public void TestCheckValidMoves(bool fullBoard)
        {
            var gameObj = new GameObject("board");
            var board = gameObj.AddComponent<StringBoard>();

            var boardString =
@"0105
                  1024
                  0203
                  2022";
            board.SetBoard(boardString);
            var boardSize = board.GetMaxBoardSize();
            if (!fullBoard)
            {
                board.CurrentRows -= 1;
            }

            var validMoves = GetValidMoves4x4(fullBoard, boardSize);

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

            // Make sure iterating over AllMoves is OK with the smaller board
            foreach (var move in board.AllMoves())
            {
                var expected = validIndices.Contains(move.MoveIndex);
                Assert.AreEqual(expected, board.IsMoveValid(move), $"({move.Row}, {move.Column}, {move.Direction})");
            }

            HashSet<int> validIndicesFromIterator = new HashSet<int>();
            foreach (var move in board.ValidMoves())
            {
                validIndicesFromIterator.Add(move.MoveIndex);
            }
            Assert.IsTrue(validIndices.SetEquals(validIndicesFromIterator));
        }
    }
}
