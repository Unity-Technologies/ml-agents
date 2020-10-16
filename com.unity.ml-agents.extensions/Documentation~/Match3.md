# Match-3 Game Support

We provide some utilities to integrate ML-Agents with Match-3 games.

## AbstractBoard class
The `AbstractBoard` is the bridge between ML-Agents and your game. It allows ML-Agents to
* ask your game what the "color" of a cell is
* ask whether the cell is a "special" piece type or not
* ask your game whether a move is allowed
* request that your game make a move

These are handled by implementing the `GetCellType()`, `IsMoveValid()`, and `MakeMove()` abstract methods.

The AbstractBoard also tracks the number of rows, columns, and potential piece types that the board can have.

#### `public abstract int GetCellType(int row, int col)`
Returns the "color" of piece at the given row and column.
This should be between 0 and NumCellTypes-1 (inclusive).
The actual order of the values doesn't matter.

#### `public abstract int GetSpecialType(int row, int col)`
Returns the special type of the piece at the given row and column.
This should be between 0 and NumSpecialTypes (inclusive).
The actual order of the values doesn't matter.

#### `public abstract bool IsMoveValid(Move m)`
Check whether the particular `Move` is valid for the game.
The actual results will depend on the rules of the game, but we provide the `SimpleIsMoveValid()` method
that handles basic match3 rules with no special or immovable pieces.

#### `public abstract bool MakeMove(Move m)`
Instruct the game to make the given move. Returns true if the move was made.
Note that during training, a move that was marked as invalid may occasionally still be
requested. If this happens, it is safe to do nothing and request another move.

## Move struct
The Move struct encapsulates a swap of two adjacent cells. You can get the number of potential moves
for a board of a given size with. `Move.NumPotentialMoves(NumRows, NumColumns)`. There are two helper
functions to create a new `Move`:
* `public static Move FromMoveIndex(int moveIndex, int maxRows, int maxCols)` can be used to
iterate over all potential moves for the board by looping from 0 to `Move.NumPotentialMoves()`
* `public static Move FromPositionAndDirection(int row, int col, Direction dir, int maxRows, int maxCols)` creates
a `Move` from a row, column, and direction (and board size).

## `Match3Sensor` and `Match3SensorComponent` classes
The `Match3Sensor` generates observations about the state using the `AbstractBoard` interface. You can
choose whether to use vector or "visual" observations; in theory, visual observations should perform
better because they are 2-dimensional like the board, but we need to experiment more on this.

A `Match3SensorComponent` generates a `Match3Sensor` at runtime, and should be added to the same GameObject
as your `Agent` implementation. You do not need to write any additional code to use them.

## `Match3Actuator` and `Match3ActuatorComponent` classes
The `Match3Actuator` converts actions from training or inference into a `Move` that is sent to` AbstractBoard.MakeMove()`
It also checks `AbstractBoard.IsMoveValid` for each potential move and uses this to set the action mask for Agent.

A `Match3ActuatorComponent` generates a `Match3Actuator` at runtime, and should be added to the same GameObject
as your `Agent` implementation.  You do not need to write any additional code to use them.

# Setting up match-3 simulation
* Implement the `AbstractBoard` methods to integrate with your game.
* Give the `Agent` rewards when it does what you want it to (match multiple pieces in a row, clears pieces of a certain
type, etc).
* Add the `Agent`, `AbstractBoard` implementation, `Match3SensorComponent`, and `Match3ActuatorComponent` to the same
`GameObject`.
* Call `Agent.RequestDecision()` when you're ready for the `Agent` to make a move on the next `Academy` step. During
the next `Academy` step, the `MakeMove()` method on the board will be called.
