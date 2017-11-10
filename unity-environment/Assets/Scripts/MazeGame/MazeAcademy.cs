using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeAcademy : Academy {

    public override void InitializeAcademy()
    {
        RestartGame();
    }


    public override void AcademyReset()
	{
        RestartGame();
    }

	public override void AcademyStep()
	{
    }

    public Vector2i mazeDimension;
    public bool randomWallChance = false;
    public float wallChanceOnNonPath = 0.3f;
    public int maxStepAllowed = 20;
    public float failureReward = -100;
    public float maxWinReward = 100;
    public float goToWallReward;
    public float goUpReward = 1;
    public float goCloserReward = 1;
    public float stepCostReward = -0.1f;
    
    private Vector2i startPosition;
    private Vector2i goalPosition;
    private Vector2i currentPlayerPosition;
    public bool Win { get; private set; }
    public float[,] map;
    private Dictionary<int, GameState> savedState;



    public readonly int WallInt = 0;
    public readonly int PlayerInt = 2;
    public readonly int PathInt = 1;
    public readonly int GoalInt = 3;

    private struct GameState
    {
        public float[,] map;
        public Vector2i startPosition;
        public Vector2i goalPosition;
        public Vector2i currentPlayerPosition;
        public bool win;
    }


    // Use this for initialization
    void Start()
    {
        savedState = new Dictionary<int, GameState>();
        RestartGame();
    }

    /// <summary>
    /// take a action and return the reward
    /// </summary>
    /// <param name="action">0 left 1 right 2 down 3 up</param>
    /// <returns>reward of this action</returns>
    public float StepAction(int action)
    {
        float returnReward = 0;
        //calculate the distance to goal before the action
        int distanceBefore = currentPlayerPosition.ManhattanDistanceTo(goalPosition);

        Vector2i toPosition = currentPlayerPosition;
        //do the action
        switch (action)
        {
            case 0:
                toPosition.x -= 1;
                break;
            case 1:
                toPosition.x += 1;
                break;
            case 2:
                toPosition.y -= 1;
                break;
            case 3:
                toPosition.y += 1;
                break;
            default:
                Debug.LogError("invalid action number");
                break;
        }

        bool reachGoal;
        float stepChangedReward;
        StepFromTo(currentPlayerPosition, toPosition, out stepChangedReward, out reachGoal);
        returnReward += stepChangedReward;

        //reward for move closer to the destination
        int distanceAfter = currentPlayerPosition.ManhattanDistanceTo(goalPosition);
        if (distanceAfter < distanceBefore)
        {
            returnReward += goCloserReward;
        }

        //reward for going up
        if (action == 3)
        {
            returnReward += goUpReward;
        }
        
        if (reachGoal)
        {
            done = true;
            Win = true;
        }
        if (currentStep >= maxStepAllowed)
        {
            done = true;
            Win = false;
            returnReward += failureReward;
        }

        return returnReward;
    }


    public void RestartGame()
    {
        RegenerateMap();
    }

    public void SaveState(int key)
    {
        float[,] copiedMap = new float[mazeDimension.x, mazeDimension.y];
        System.Buffer.BlockCopy(map, 0, copiedMap, 0, map.Length * sizeof(float));

        GameState state = new GameState();
        state.map = copiedMap;
        state.currentPlayerPosition = currentPlayerPosition;
        state.goalPosition = goalPosition;
        state.startPosition = startPosition;
        state.win = Win;
        savedState[key] = state;
    }


    public bool LoadState(int key)
    {
        MazeAcademy fromEnv = this;
        if (fromEnv.savedState.ContainsKey(key))
        {
            GameState state = fromEnv.savedState[key];
            System.Buffer.BlockCopy(state.map, 0, map, 0, map.Length * sizeof(float));
            currentPlayerPosition = state.currentPlayerPosition;
            goalPosition = state.goalPosition;
            startPosition = state.startPosition;
            Win = state.win;
            return true;
        }
        else
        {
            return false;
        }
    }



    /// <summary>
    /// get the state of the map. a array of all the blocks. 1 means road, 2 means blocked 3 means goal 4 means player; 
    /// totally 6*6 = 36 floats
    /// </summary>
    /// <returns></returns>
    public float[] GetState()
    {
        float[] result = new float[mazeDimension.x * mazeDimension.y];
        for (int x = 0; x < mazeDimension.x; ++x)
        {
            for (int y = 0; y < mazeDimension.y; ++y)
            {
                result[y + x * mazeDimension.y] = map[x, y];
            }
        }

        return result;
    }


    private void RegenerateMap()
    {
        map = new float[mazeDimension.x, mazeDimension.y];
        GeneratePossiblePath();
        GenerateExtraPath();
    }

    //mark a path with true. The generator will guarantee that this path is walkable
    private void GeneratePossiblePath()
    {


        int place = Random.Range(0, mazeDimension.x);
        int prevPlace = place;
        map[place, mazeDimension.y - 1] = GoalInt;
        goalPosition = new Vector2i(place, mazeDimension.y - 1);

        bool toggle = true;
        for (int i = mazeDimension.y - 2; i >= 0; --i)
        {
            if (toggle)
            {
                toggle = false;
                map[prevPlace, i] = PathInt;
            }
            else
            {
                toggle = true;
                place = Random.Range(0, mazeDimension.x);
                for (int j = Mathf.Min(place, prevPlace); j <= Mathf.Max(place, prevPlace); ++j)
                {
                    map[j, i] = PathInt;
                }
            }
            prevPlace = place;
        }

        startPosition = new Vector2i(place, 0);
        map[place, 0] = PlayerInt;
        currentPlayerPosition = startPosition;
    }

    private void GenerateExtraPath()
    {
        float wallChance = wallChanceOnNonPath;
        if (randomWallChance)
        {
            wallChance = Random.Range(0.0f, 1.0f);
        }
        for (int i = 0; i < mazeDimension.x; ++i)
        {
            for (int j = 0; j < mazeDimension.y; ++j)
            {
                if (map[i, j] == WallInt && Random.Range(0.0f, 1.0f) > wallChance)
                {
                    map[i, j] = PathInt;
                }
            }
        }
    }

    private void StepFromTo(Vector2i from, Vector2i to, out float stepChangedReward, out bool reachedGoal)
    {
        Debug.Assert(map[from.x, from.y] == PlayerInt && currentPlayerPosition.Equals(from));
        stepChangedReward = 0;
        stepChangedReward += stepCostReward;
        if (to.x < 0 || to.y < 0 || to.x >= mazeDimension.x || to.y >= mazeDimension.y)
        {
            //run to the edge
            stepChangedReward += goToWallReward;
            reachedGoal = false;
        }
        else
        {
            if (map[to.x, to.y] == WallInt)
            {
                //run into a wall
                //run to the edge
                stepChangedReward += goToWallReward;
                reachedGoal = false;
            }
            else if (map[to.x, to.y] == GoalInt)
            {
                //reach the goal
                stepChangedReward += maxWinReward;
                reachedGoal = true;
            }
            else
            {
                //move successfully
                map[currentPlayerPosition.x, currentPlayerPosition.y] = PathInt;
                currentPlayerPosition = to;
                map[currentPlayerPosition.x, currentPlayerPosition.y] = PlayerInt;
                reachedGoal = false;
            }
        }
    }
}
