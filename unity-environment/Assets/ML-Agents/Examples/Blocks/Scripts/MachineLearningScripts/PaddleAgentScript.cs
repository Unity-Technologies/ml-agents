using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleAgentScript : Agent {
    [Header("Specific to Breakout")]
    public GameObject ball;
    public PaddleScript paddle;
    public GameManager gm;

    Vector3 paddleStart;
    Vector3 ballStart;

    private void Awake()
    {
        paddleStart = paddle.transform.position;
        ballStart = ball.transform.position;
    }

    float timeBetweenHits = 0f;
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Ball" && timeBetweenHits > 0f)
        {
            gm.BallBounce();
            timeBetweenHits = 0f;
        }
    }

    private void Update()
    {
        timeBetweenHits += Time.deltaTime;
    }

    public void SetDone(float rewardValue)
    {
        reward += rewardValue;
        done = true;
    }

    public void AddReward(float rewardValue)
    {
        reward += rewardValue;
    }

    enum direction { left, right, still };
    // to be implemented by the developer
    public override void AgentStep(float[] act)
    {
        direction dir = direction.still;
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            float leftAmount = act[0];
            float rightAmount = act[1];
            if (leftAmount > 1f) dir = direction.left;
            if (rightAmount > 1f) dir = direction.right;

            if (dir == direction.left) paddle.Left();
            if (dir == direction.right) paddle.Right();

            if (done == false) reward += gm.timerLoss * Time.deltaTime;
        }
        else
        {
            var left = (int)act[0];
            var right = (int)act[0];
            if (left != 0 || right != 0)
            {
                if (left == 1) dir = direction.right;
                if (right == 2) dir = direction.left;

                if (dir == direction.left) paddle.Left();
                if (dir == direction.right) paddle.Right();
            }
            if (done == false)
            {
                if (done == false) reward += gm.timerLoss * Time.deltaTime;
            }
        }
    }

    public List<float> getFloatsXy(Vector3 target, float normDivisor)
    {
        var result = new List<float>();
        result.Add(target.x / normDivisor);
        result.Add(target.y / normDivisor);
        return result;
    }

    public override List<float> CollectState()
    {
        var state = new List<float>();

        var ballLoc = ball.transform.position - gameObject.transform.position;
        state.AddRange(getFloatsXy(ballLoc, 4f));

        var ballVel = ball.transform.GetComponent<Rigidbody>().velocity;
        state.AddRange(getFloatsXy(ballVel, 2f));

        var paddleLoc = paddle.transform.position - paddleStart;
        state.AddRange(getFloatsXy(paddleLoc, 2.5f));

        return state;
    }

    // to be implemented by the developer
    public override void AgentReset()
    {
        if (gm == null) gm = GetComponentInParent<GameManager>();
        gm.score = 0;
        var blocks = gm.GetComponentsInChildren<BlockScript>();
        foreach (var b in blocks) b.Respawn();
        paddle.transform.position = paddleStart;
        ball.GetComponent<BallScript>().RandomStartPosition();
        ball.GetComponent<Rigidbody>().velocity = Vector3.zero;
        ball.SetActive(true);
    }
}
