using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TennisAcademy : Academy
{

    [Header("Specific to Tennis")]
    public GameObject ball;

    public override void AcademyReset()
    {
        float ballOut = Random.Range(4f, 11f);
        int flip = Random.Range(0, 2);
        if (flip == 0)
        {
            ball.transform.position = new Vector3(-ballOut, 5f, 5f);
        }
        else
        {
            ball.transform.position = new Vector3(ballOut, 5f, 5f);
        }
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        ball.transform.localScale = new Vector3(1, 1, 1) * resetParameters["ballSize"];
    }

    public override void AcademyStep()
    {

    }

}
