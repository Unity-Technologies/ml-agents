using Unity.MLAgents.Extensions.Match3;

namespace Unity.MLAgentsExamples
{


    public class Match3Board : AbstractBoard
    {
        public int RandomSeed = -1;

        public const int k_EmptyCell = -1;

        int[,] m_Cells;
        bool[,] m_Matched;

        System.Random m_Random;

        void Awake()
        {
            m_Cells = new int[Columns, Rows];
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
            return m_Cells[col, row];
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
                        if (m_Cells[j, i] != m_Cells[j, iOffset])
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
                            // TODO check whether already matched for scoring
                            m_Matched[j, i + k] = true;
                        }
                    }

                    // Check vertically
                    var matchedCols = 0;
                    for (var jOffset = j; jOffset < Columns; jOffset++)
                    {
                        if (m_Cells[j, i] != m_Cells[jOffset, i])
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
                            // TODO check whether already matched for scoring
                            m_Matched[j + k, i] = true;
                        }
                    }
                }
            }

            return madeMatch;
        }

        public int ClearMatchedCells()
        {
            int numMatchedCells = 0;
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    if (m_Matched[j, i])
                    {
                        numMatchedCells++;
                        m_Cells[j, i] = k_EmptyCell;
                    }
                }
            }

            ClearMarked(); // TODO clear here or at start of matching?
            return numMatchedCells;
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
                    if (m_Cells[j, readIndex] != k_EmptyCell)
                    {
                        writeIndex++;
                    }
                }

                // Fill in empties at the end
                // TODO combine with random drops?
                for (; writeIndex < Rows; writeIndex++)
                {
                    madeChanges = true;
                    m_Cells[j, writeIndex] = k_EmptyCell;
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
                    if (m_Cells[j, i] == k_EmptyCell)
                    {
                        madeChanges = true;
                        m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
                    }
                }
            }

            return madeChanges;
        }

        public int[,] Cells
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
                    m_Cells[j, i] = m_Random.Next(0, NumCellTypes);
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


    }
}
