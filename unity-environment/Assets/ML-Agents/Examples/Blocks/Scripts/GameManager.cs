using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

[Serializable]
public class UnityFloatEvent : UnityEvent<float> { };

public class GameManager : MonoBehaviour {

    public float score;
    public float brickHitScore = 0.1f;
    public float lostBallScore = -10;
    public float timerLoss = -0.1f;
    public float ballBounceScore = 1f;


    public UnityFloatEvent OnBrickHit;
    public UnityFloatEvent OnBallLost;
    public UnityFloatEvent OnBallBounce;

    // Use this for initialization
    void Start () {
        Time.captureFramerate = 60;
        score = 0;
	}

    private void Update()
    {
        score += timerLoss * Time.deltaTime;
    }

    public void BrickHit()
    {
        score += brickHitScore;
        if (OnBrickHit != null) OnBrickHit.Invoke(brickHitScore);
    }

    public void BallLost()
    {
        score += lostBallScore;
        if (OnBallLost != null) OnBallLost.Invoke(lostBallScore);
    }

    public void BallBounce()
    {
        score += ballBounceScore;
        if (OnBallBounce != null) OnBallBounce.Invoke(ballBounceScore);
    }
	
}
