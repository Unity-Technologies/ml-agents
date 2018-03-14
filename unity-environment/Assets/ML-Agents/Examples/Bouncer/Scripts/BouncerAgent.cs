using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BouncerAgent : Agent {

    [Header("Bouncer Specific")]
    public GameObject banana;
    public GameObject bodyObject;
    Rigidbody rb;
    Vector3 lookDir;
    public float strength = 10f;
    float jumpCooldown = 0f;
    int numberJumps = 20;
    int jumpLeft = 20;

    public override void InitializeAgent()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        lookDir = Vector3.zero;
    }

    public override void CollectObservations()
    {
        AddVectorObs(gameObject.transform.localPosition);
        AddVectorObs(banana.transform.localPosition);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        float x = Mathf.Clamp(vectorAction[0], -1, 1);
        float y = Mathf.Clamp(vectorAction[1], 0, 1);
        float z = Mathf.Clamp(vectorAction[2], -1, 1);
        rb.AddForce( new Vector3(x, y+1, z) *strength);

        AddReward(-0.05f * (
            vectorAction[0] * vectorAction[0] +
            vectorAction[1] * vectorAction[1] +
            vectorAction[2] * vectorAction[2]) / 3f);

        lookDir = new Vector3(x, y, z);
    }

    public override void AgentReset()
    {

        gameObject.transform.localPosition = new Vector3(
            (1 - 2 * Random.value) *5, 2, (1 - 2 * Random.value)*5);
        rb.velocity = default(Vector3);
        GameObject environment = gameObject.transform.parent.gameObject;
        BouncerBanana[] bananas = 
            environment.GetComponentsInChildren<BouncerBanana>();
        foreach (BouncerBanana bb in bananas)
        {
            bb.Respawn();
        }
        jumpLeft = numberJumps;
    }

    public override void AgentOnDone()
    {

    }

    private void FixedUpdate()
    {
        if ((Physics.Raycast(transform.position, new Vector3(0f,-1f,0f), 0.51f))
            && jumpCooldown <= 0f)
        {
            RequestDecision();
            jumpLeft -= 1;
            jumpCooldown = 0.1f;
            rb.velocity = default(Vector3);
        }
        jumpCooldown -= Time.fixedDeltaTime;
        if (gameObject.transform.position.y < -1)
        {
            AddReward(-1);
            Done();
            return;
        }
        if ((gameObject.transform.localPosition.x < -19)
            ||(gameObject.transform.localPosition.x >19)
            || (gameObject.transform.localPosition.z < -19)
            || (gameObject.transform.localPosition.z > 19)
           )
        {
            AddReward(-1);
            Done();
            return;
        }
        if (jumpLeft == 0)
        {
            Done();
        }

        bodyObject.transform.rotation = Quaternion.Lerp(bodyObject.transform.rotation,
                                  Quaternion.LookRotation(lookDir),
                                  Time.fixedDeltaTime * 10f);

    }
}
