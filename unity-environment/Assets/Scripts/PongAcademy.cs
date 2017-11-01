using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongAcademy : Academy {

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
        StepAction(agentLeft.nextAction, agentRight.nextAction);
        agentLeft.currentState = GetState(0);
        agentRight.currentState = GetState(1);
        agentLeft.reward = RewardsLastStep[0];
        agentRight.reward = RewardsLastStep[1];
    }

    public PongAgent agentLeft;
    public PongAgent agentRight;



    //------------------old game logic------------------------

    public float failureReward = -1;
    public float winReward = 1;
    public float hitBallReward = 0.1f;

    public float racketSpeed = 0.02f;
    public float ballSpeed = 0.01f;
    public float racketWidth = 0.05f;

    public readonly int ActionUp = 1;
    public readonly int ActionDown = -1;
    public readonly int ActionStay = 0;

    public float leftStartX = -1;
    public float rightStartX = 1;
    public Vector2 arenaSize = new Vector2(2.2f, 1.0f);

    public GameState CurrentGameState { get { return currentGameState; } }
    private GameState currentGameState;


    public int GameWinPlayer
    {
        get
        {
            return currentGameState.gameWinPlayer;
        }
        protected set
        {
            currentGameState.gameWinPlayer = value;
        }
    }
    public float[] RewardsLastStep
    {
        get
        {
            return new float[] { currentGameState.rewardLastStepLeft, currentGameState.rewardLastStepRight };
        }
    }


    public struct GameState
    {
        public Vector2 ballVelocity;
        public Vector2 ballPosition;
        public float leftY;
        public float rightY;
        public bool gameHasEnded;
        public int gameWinPlayer;
        public float rewardLastStepLeft;
        public float rewardLastStepRight;
    }

   
    /// <summary>
    /// take actions and step the environment
    /// </summary>
    /// <param name="action">some actions for each player. 0 is down, 1 is not moving, 2 is up</param>
    public void StepAction(params int[] action)
    {
        //clear the reward 
        currentGameState.rewardLastStepLeft = 0;
        currentGameState.rewardLastStepRight = 0;
        //use AI if the action is -1
        if (action[0] <= -1)
        {
            float y = currentGameState.leftY;
            if (y > currentGameState.ballPosition.y)
            {
                action[0] = 0;
            }
            else
            {
                action[0] = 2;
            }
        }
        if (action[1] <= -1)
        {
            float y = currentGameState.rightY;
            if (y > currentGameState.ballPosition.y)
            {
                action[1] = 0;
            }
            else
            {
                action[1] = 2;
            }
        }
        //move the rackets
        currentGameState.leftY += racketSpeed * (action[0] - 1);
        currentGameState.leftY = Mathf.Clamp(currentGameState.leftY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);
        currentGameState.rightY += racketSpeed * (action[1] - 1);
        currentGameState.rightY = Mathf.Clamp(currentGameState.rightY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);

        //move the ball
        Vector2 oldBallPosition = currentGameState.ballPosition;
        currentGameState.ballPosition += currentGameState.ballVelocity;

        //detect collision of ball with wall
        Vector2 newBallVel = currentGameState.ballVelocity;
        if (currentGameState.ballPosition.y > arenaSize.y / 2 || currentGameState.ballPosition.y < -arenaSize.y / 2)
        {
            newBallVel.y = -newBallVel.y;

        }
        if (currentGameState.ballPosition.x > arenaSize.x / 2)
        {
            currentGameState.rewardLastStepLeft += winReward;
            currentGameState.rewardLastStepRight += failureReward;
            GameWinPlayer = 0;
        }
        else if (currentGameState.ballPosition.x < -arenaSize.x / 2)
        {
            currentGameState.rewardLastStepRight += winReward;
            currentGameState.rewardLastStepLeft += failureReward;
            GameWinPlayer = 1;
        }

        //detect collision of the ball with the rackets
        if (currentGameState.ballPosition.x < leftStartX && oldBallPosition.x > leftStartX)
        {
            Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
            float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - leftStartX) / moveVector.x) + oldBallPosition).y;
            float yHitRatio = (currentGameState.leftY - yHit) / (racketWidth / 2);
            if (Mathf.Abs(yHitRatio) < 1)
            {
                //hit the left racket
                newBallVel.x = -newBallVel.x;
                newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                newBallVel = newBallVel.normalized * ballSpeed;
                currentGameState.rewardLastStepLeft += hitBallReward;
                agentLeft.hitRate100.AddValue(1);
            }
            agentLeft.hitRate100.AddValue(0);
        }
        else if (currentGameState.ballPosition.x > rightStartX && oldBallPosition.x < rightStartX)
        {
            Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
            float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - rightStartX) / moveVector.x) + oldBallPosition).y;
            float yHitRatio = (currentGameState.rightY - yHit) / (racketWidth / 2);
            if (Mathf.Abs(yHitRatio) < 1)
            {
                //hit the right racket
                newBallVel.x = -newBallVel.x;
                newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                newBallVel = newBallVel.normalized * ballSpeed;
                currentGameState.rewardLastStepRight += hitBallReward;
                agentRight.hitRate100.AddValue(1);
            }
            agentRight.hitRate100.AddValue(0);
        }

        //update the velocity
        currentGameState.ballVelocity = newBallVel;
        
    }


    public void RestartGame()
    {
        currentGameState.leftY = 0;
        currentGameState.rightY = 0;
        currentGameState.ballPosition = Vector2.zero;
        Vector2 initialVel = Random.insideUnitCircle;
        if (Mathf.Abs(initialVel.y) > Mathf.Abs(initialVel.x))
        {
            float temp = initialVel.y;
            initialVel.y = initialVel.x;
            initialVel.x = temp;
        }
        currentGameState.ballVelocity = initialVel.normalized * ballSpeed;
        currentGameState.rewardLastStepRight = 0;
        currentGameState.rewardLastStepLeft = 0;
        currentGameState.gameWinPlayer = -1;

        agentLeft.currentState = GetState(0);
        agentRight.currentState = GetState(1);
        agentLeft.reward = RewardsLastStep[0];
        agentRight.reward = RewardsLastStep[1];
    }


    /// <summary>
    /// get a array of float that represent the current state
    /// </summary>
    /// <param name="type">what type of state.One game might have different representation of its state. 0 means for left player, 1 means for right</param>
    /// <returns></returns>
    public float[] GetState(int type = 0)
    {
        float[] result = null;
        if (type == 0)
        {
            result = new float[] {
                currentGameState.leftY,
                currentGameState.rightY,
                currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        else
        {
            result = new float[] {
                currentGameState.rightY,
                currentGameState.leftY,
                -currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                -currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        return result;
    }
}
