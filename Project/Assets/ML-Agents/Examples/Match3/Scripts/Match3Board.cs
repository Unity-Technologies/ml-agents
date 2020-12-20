using Unity.MLAgents.Extensions.Match3;
using UnityEngine;

namespace Unity.MLAgentsExamples
{


    public class Match3Board : AbstractBoard
    {
        public int RandomSeed = -1;

        public const int k_EmptyCell = -1;
        [Tooltip("Points earned for clearing a basic cell (cube)")]
        public int BasicCellPoints = 1;

        [Tooltip("Points earned for clearing a special cell (sphere)")]
        public int SpecialCell1Points = 2;

        [Tooltip("Points earned for clearing an extra special cell (plus)")]
        public int SpecialCell2Points = 3;

        (int, int)[,] m_Cells;
        bool[,] m_Matched;

        System.Random m_Random;

        void Awake()
        {
            m_Cells = new (int, int)[Columns, Rows];
            m_Matched = new bool[Columns, Rows];

            m_Random = new System.Random(RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed);

            InitRandom();
        }

        public override bool MakeMove(Move move)
        {
            if (!IsMoveValid(move))
            {
                return false;
            }
            var originalValue = m_Cells[move.Column, move.Row];
            var (otherRow, otherCol) = move.OtherCell();
            var destinationValue = m_Cells[otherCol, otherRow];

            m_Cells[move.Column, move.Row] = destinationValue;
            m_Cells[otherCol, otherRow] = originalValue;
            return true;
        }

        public override int GetCellType(int row, int col)
        {
            return m_Cells[col, row].Item1;
        }

        public override int GetSpecialType(int row, int col)
        {
            return m_Cells[col, row].Item2;
        }

        public override bool IsMoveValid(Move m)
        {
            if (m_Cells == null)
            {
                return false;
            }

            return SimpleIsMoveValid(m);
        }

        public bool MarkMatchedCells(int[,] cells = null)
        {
            ClearMarked();
            bool madeMatch = false;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    // Check vertically
                    var matchedRows = 0;
                    for (var iOffset = i; iOffset < Rows; iOffset++)
                    {
                        if (m_Cells[j, i].Item1 != m_Cells[j, iOffset].Item1)
                        {
                            break;
                        }

                        matchedRows++;
                    }

                    if (matchedRows >= 3)
                    {
                        madeMatch = true;
                        for (var k = 0; k < matchedRows; k++)
                        {
                            m_Matched[j, i + k] = true;
                        }
                    }

                    // Check vertically
                    var matchedCols = 0;
                    for (var jOffset = j; jOffset < Columns; jOffset++)
                    {
                        if (m_Cells[j, i].Item1 != m_Cells[jOffset, i].Item1)
                        {
                            break;
                        }

                        matchedCols++;
                    }

                    if (matchedCols >= 3)
                    {
                        madeMatch = true;
                        for (var k = 0; k < matchedCols; k++)
                        {
                            m_Matched[j + k, i] = true;
                        }
                    }
                }
            }

            return madeMatch;
        }

        /// <summary>
        /// Sets cells that are matched to the empty cell, and returns the score earned.
        /// </summary>
        /// <returns></returns>
        public int ClearMatchedCells()
        {
            var pointsByType = new[] { BasicCellPoints, SpecialCell1Points, SpecialCell2Points };
            int pointsEarned = 0;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Matched[j, i])
                    {
                        var speciaType = GetSpecialType(i, j);
                        pointsEarned += pointsByType[speciaType];
                        m_Cells[j, i] = (k_EmptyCell, 0);
                    }
                }
            }

            ClearMarked(); // TODO clear here or at start of matching?
            return pointsEarned;
        }

        public bool DropCells()
        {
            var madeChanges = false;
            // Gravity is applied in the negative row direction
            for (var j = 0; j < Columns; j++)
            {
                var writeIndex = 0;
                for (var readIndex = 0; readIndex < Rows; readIndex++)
                {
                    m_Cells[j, writeIndex] = m_Cells[j, readIndex];
                    if (m_Cells[j, readIndex].Item1 != k_EmptyCell)
                    {
                        writeIndex++;
                    }
                }

                // Fill in empties at the end
                for (; writeIndex < Rows; writeIndex++)
                {
                    madeChanges = true;
                    m_Cells[j, writeIndex] = (k_EmptyCell, 0);
                }
            }

            return madeChanges;
        }

        public bool FillFromAbove()
        {
            bool madeChanges = false;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Cells[j, i].Item1 == k_EmptyCell)
                    {
                        madeChanges = true;
                        m_Cells[j, i] = (GetRandomCellType(), GetRandomSpecialType());
                    }
                }
            }

            return madeChanges;
        }

        public (int, int)[,] Cells
        {
            get { return m_Cells; }
        }

        public bool[,] Matched
        {
            get { return m_Matched; }
        }

        // Initialize the board to random values.
        public void InitRandom()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m_Cells[j, i] = (GetRandomCellType(), GetRandomSpecialType());
                }
            }
        }

        public void InitSettled()
        {
            InitRandom();
            while (true)
            {
                var anyMatched = MarkMatchedCells();
                if (!anyMatched)
                {
                    return;
                }
                ClearMatchedCells();
                DropCells();
                FillFromAbove();
            }
        }

        void ClearMarked()
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m_Matched[j, i] = false;
                }
            }
        }

        int GetRandomCellType()
        {
            return m_Random.Next(0, NumCellTypes);
        }

        int GetRandomSpecialType()
        {
            // 1 in N chance to get a type-2 special
            // 2 in N chance to get a type-1 special
            // otherwise 0 (boring)
            var N = 10;
            var val = m_Random.Next(0, N);
            if (val == 0)
            {
                return 2;
            }

            if (val <= 2)
            {
                return 1;
            }

            return 0;
        }

    }
}
