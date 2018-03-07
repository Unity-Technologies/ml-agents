// Put this script on your blue cube.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HallwayAgent : Agent
{
    public GameObject ground; // ground game object. we will use the area bounds to spawn the blocks
    public GameObject area;

    public GameObject goalA;
    public GameObject goalB;
    public GameObject orangeBlock; // the orange block we are going to be pushing
    public GameObject violetBlock;
    Rigidbody shortBlockRB;  // cached on initialization
    Rigidbody agentRB;  // cached on initialization
    Material groundMaterial; // cached on Awake()
    Renderer groundRenderer;
    HallwayAcademy academy;

    int selection;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        academy = FindObjectOfType<HallwayAcademy>();
        brain = FindObjectOfType<Brain>(); // only one brain in the scene so this should find our brain. BRAAAINS.

        agentRB = GetComponent<Rigidbody>(); // cache the agent rigidbody
        groundRenderer = ground.GetComponent<Renderer>(); // get the ground renderer so we can change the material when a goal is scored
        groundMaterial = groundRenderer.material; // starting material

    }

    public void RayPerception(float rayDistance,
                                 float[] rayAngles, string[] detectableObjects, float height)
    {
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(GiveCatersian(rayDistance, noisyAngle));
            position.y = height;
            Debug.DrawRay(transform.position, position, Color.red, 0.1f, true);
            RaycastHit hit;
            float[] subList = new float[detectableObjects.Length + 2];
            if (Physics.SphereCast(transform.position, 1.0f, position, out hit, rayDistance))
            {
                for (int i = 0; i < detectableObjects.Length; i++)
                {
                    if (hit.collider.gameObject.CompareTag(detectableObjects[i]))
                    {
                        subList[i] = 1;
                        subList[detectableObjects.Length + 1] = hit.distance / rayDistance;
                        break;
                    }
                }
            }
            else
            {
                subList[detectableObjects.Length] = 1f;
            }
            foreach (float f in subList)
                AddVectorObs(f);
        }
    }

    public Vector3 GiveCatersian(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian(angle));
        float z = radius * Mathf.Sin(DegreeToRadian(angle));
        return new Vector3(x, 1f, z);
    }

    public float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }


    public override void CollectObservations()
    {
        float rayDistance = 8.5f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f };
        string[] detectableObjects = { "goal", "orangeBlock", "redBlock", "wall" };
        RayPerception(rayDistance, rayAngles, detectableObjects, 0f);
    }

    // swap ground material, wait time seconds, then swap back to the regular ground material.
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        groundRenderer.material = mat;
        yield return new WaitForSeconds(time); // wait for 2 sec
        groundRenderer.material = groundMaterial;
    }


    public void MoveAgent(float[] act)
    {

        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;


        // If we're using Continuous control you will need to change the Action
        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
        }
        else
        {
            int action = Mathf.FloorToInt(act[0]);
            if (action == 0)
            {
                dirToGo = transform.forward * 1f;
            }
            else if (action == 1)
            {
                dirToGo = transform.forward * -1f;
            }
            else if (action == 2)
            {
                rotateDir = transform.up * 1f;
            }
            else if (action == 3)
            {
                rotateDir = transform.up * -1f;
            }
        }
        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange); // GO
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        AddReward(-0.0003f);

        MoveAgent(vectorAction); //perform agent actions
        bool fail = false;  // did the agent or block get pushed off the edge?

        if (!Physics.Raycast(agentRB.position, Vector3.down, 20)) // if the agent has gone over the edge, we done.
        {
            fail = true; // fell off bro
            AddReward(-1f); // BAD AGENT
                          //transform.position =  GetRandomSpawnPos(agentSpawnAreaBounds, agentSpawnArea);
            Done(); // if we mark an agent as done it will be reset automatically. AgentReset() will be called.
        }

        if (fail)
        {
            StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, .5f)); // swap ground material to indicate fail
        }
    }

    // detect when we touch the goal
    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("goal")) // touched goal
        {
            if ((selection == 0 && col.gameObject.name == "GoalA") || (selection == 1 && col.gameObject.name == "GoalB"))
            {
                AddReward(1f); // you get 5 points
                StartCoroutine(GoalScoredSwapGroundMaterial(academy.goalScoredMaterial, 2)); // swap ground material for a bit to indicate we scored.
            }
            else
            {
                AddReward(-0.1f); // you lose a point
                StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, .5f)); // swap ground material to indicate fail
            }
            Done(); // if we mark an agent as done it will be reset automatically. AgentReset() will be called.
        }
    }

    // In the editor, if "Reset On Done" is checked then AgentReset() will be called automatically anytime we mark done = true in an agent script.
    public override void AgentReset()
    {
        selection = Random.Range(0, 2);
        if (selection == 0)
        {
            orangeBlock.transform.position = new Vector3(0f + Random.Range(-3f, 3f), 2f, -15f + Random.Range(-5f, 5f)) + ground.transform.position;
            violetBlock.transform.position = new Vector3(0f, -1000f, -15f + Random.Range(-5f, 5f)) + ground.transform.position;
        }
        else
        {
            orangeBlock.transform.position = new Vector3(0f, -1000f, -15f + Random.Range(-5f, 5f)) + ground.transform.position;
            violetBlock.transform.position = new Vector3(0f, 2f, -15f + Random.Range(-5f, 5f)) + ground.transform.position;
        }
        transform.position = new Vector3(0f+ Random.Range(-3f, 3f), 1f, 0f + Random.Range(-5f, 5f)) + ground.transform.position;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        agentRB.velocity *= 0f;
    }
}

