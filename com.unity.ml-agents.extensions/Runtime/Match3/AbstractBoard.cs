namespace Unity.MLAgents.Extensions.Match3
{
    public abstract class AbstractBoard
    {
        public readonly int Rows;
        public readonly int Columns;
        public readonly int NumCellTypes;

        public AbstractBoard(int rows, int cols, int numCellTypes)
        {
            Rows = rows;
            Columns = cols;
            NumCellTypes = numCellTypes;
        }

        public abstract bool MakeMove(Move m);
        public abstract bool IsMoveValid(Move m);
        // TODO handle "special" cell types?
        public abstract int GetCellType(int row, int col);
    }
}
