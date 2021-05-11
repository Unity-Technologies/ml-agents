# Match-3 with ML-Agents

<img src="images/match3.png" align="center" width="3000"/>

## Getting started
The C# code for Match-3 exists inside of the Unity package (`com.unity.ml-agents`).
The good first step would be to take a look at how we have implemented the C# code in the example Match-3 scene (located
under /Project/Assets/ML-Agents/Examples/match3). Once you have some familiarity, then the next step would be to
implement the C# code for Match-3 from the extensions package.

Additionally, see below for additional technical specifications on the C# code for Match-3. Please note the Match-3 game
isn't human playable as implemented and can be only played via training.

## Technical specifications for Match-3 with ML-Agents

### AbstractBoard class
The `AbstractBoard` is the bridge between ML-Agents and your game. It allows ML-Agents to
* ask your game what the current and maximum sizes (rows, columns, and potential piece types) of the board are
* ask your game what the "color" of a cell is
* ask whether the cell is a "special" piece type or not
* ask your game whether a move is allowed
* request that your game make a move

These are handled by implementing the abstract methods of `AbstractBoard`.

##### `public abstract BoardSize GetMaxBoardSize()`
Returns the largest `BoardSize` that the game can use. This is used to determine the sizes of observations and sensors,
so don't make it larger than necessary.

##### `public virtual BoardSize GetCurrentBoardSize()`
Returns the current size of the board. Each field on this BoardSize must be less than or equal to the corresponding
field returned by `GetMaxBoardSize()`. This method is optional; if your always use the same size board, you don't
need to override it.

If the current board size is smaller than the maximum board size, `GetCellType()` and `GetSpecialType()` will not be
called for cells outside the current board size, and `IsValidMove` won't be called for moves that would go outside of
the current board size.

##### `public abstract int GetCellType(int row, int col)`
Returns the "color" of piece at the given row and column.
This should be between 0 and BoardSize.NumCellTypes-1 (inclusive).
The actual order of the values doesn't matter.

##### `public abstract int GetSpecialType(int row, int col)`
Returns the special type of the piece at the given row and column.
This should be between 0 and BoardSize.NumSpecialTypes (inclusive).
The actual order of the values doesn't matter.

##### `public abstract bool IsMoveValid(Move m)`
Check whether the particular `Move` is valid for the game.
The actual results will depend on the rules of the game, but we provide the `SimpleIsMoveValid()` method
that handles basic match3 rules with no special or immovable pieces.

##### `public abstract bool MakeMove(Move m)`
Instruct the game to make the given move. Returns true if the move was made.
Note that during training, a move that was marked as invalid may occasionally still be
requested. If this happens, it is safe to do nothing and request another move.

### `Move` struct
The Move struct encapsulates a swap of two adjacent cells. You can get the number of potential moves
for a board of a given size with. `Move.NumPotentialMoves(maxBoardSize)`. There are two helper
functions to create a new `Move`:
* `public static Move FromMoveIndex(int moveIndex, BoardSize maxBoardSize)` can be used to
iterate over all potential moves for the board by looping from 0 to `Move.NumPotentialMoves()`
* `public static Move FromPositionAndDirection(int row, int col, Direction dir, BoardSize maxBoardSize)` creates
a `Move` from a row, column, and direction (and board size).

### `BoardSize` struct
Describes the "size" of the board, including the number of potential piece types that the board can have.
This is returned by the AbstractBoard.GetMaxBoardSize() and GetCurrentBoardSize() methods.

#### `Match3Sensor` and `Match3SensorComponent` classes
The `Match3Sensor` generates observations about the state using the `AbstractBoard` interface. You can
choose whether to use vector or "visual" observations; in theory, visual observations should perform
better because they are 2-dimensional like the board, but we need to experiment more on this.

A `Match3SensorComponent` generates `Match3Sensor`s (the exact number of sensors depends on your configuration)
at runtime, and should be added to the same GameObject as your `Agent` implementation. You do not need to write any
additional code to use them.

#### `Match3Actuator` and `Match3ActuatorComponent` classes
The `Match3Actuator` converts actions from training or inference into a `Move` that is sent to` AbstractBoard.MakeMove()`
It also checks `AbstractBoard.IsMoveValid` for each potential move and uses this to set the action mask for Agent.

A `Match3ActuatorComponent` generates a `Match3Actuator` at runtime, and should be added to the same GameObject
as your `Agent` implementation.  You do not need to write any additional code to use them.

### Setting up Match-3 simulation
* Implement the `AbstractBoard` methods to integrate with your game.
* Give the `Agent` rewards when it does what you want it to (match multiple pieces in a row, clears pieces of a certain
type, etc).
* Add the `Agent`, `AbstractBoard` implementation, `Match3SensorComponent`, and `Match3ActuatorComponent` to the same
`GameObject`.
* Call `Agent.RequestDecision()` when you're ready for the `Agent` to make a move on the next `Academy` step. During
the next `Academy` step, the `MakeMove()` method on the board will be called.

## Implementation Details

### Action Space
The indexing for actions is the same as described in
[Human Like Playtesting with Deep Learning](https://www.researchgate.net/publication/328307928_Human-Like_Playtesting_with_Deep_Learning)
(for example, Figure 2b). The horizontal moves are enumerated first, then the vertical ones.
<img src="images/match3-moves.png" align="center"/>

## Feedback
If you are a Match-3 developer and are trying to leverage ML-Agents for this scenario,
[we want to hear from you](https://forms.gle/TBsB9jc8WshgzViU9). Additionally, we are also looking for interested
Match-3 teams to speak with us for 45 minutes. If you are interested, please indicate that in the
[form](https://forms.gle/TBsB9jc8WshgzViU9). If selected, we will provide gift cards as a token of appreciation.
