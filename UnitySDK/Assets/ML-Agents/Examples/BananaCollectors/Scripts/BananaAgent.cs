using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class BananaAgent : Agent
{
    private BananaAcademy myAcademy;
    public GameObject area;
    BananaArea myArea;
    bool frozen;
    bool poisioned;
    bool satiated;
    bool shoot;
    float frozenTime;
    float effectTime;
    Rigidbody agentRb;
    private int bananas;

    // Speed of agent rotation.
    public float turnSpeed = 300;

    // Speed of agent movement.
    public float moveSpeed = 2;
    public Material normalMaterial;
    public Material badMaterial;
    public Material goodMaterial;
    public Material frozenMaterial;
    public GameObject myLaser;
    public bool contribute;
    private RayPerception rayPer;
    public bool useVectorObs;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        agentRb = GetComponent<Rigidbody>();
        Monitor.verticalOffset = 1f;
        myArea = area.GetComponent<BananaArea>();
        rayPer = GetComponent<RayPerception>();
        myAcademy = FindObjectOfType<BananaAcademy>();
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            float rayDistance = 50f;
            float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
            string[] detectableObjects = { "banana", "agent", "wall", "badBanana", "frozenAgent" };
            AddVectorObs(rayPer.Perceive(rayDistance, rayAngles, detectableObjects, 0f, 0f));
            Vector3 localVelocity = transform.InverseTransformDirection(agentRb.velocity);
            AddVectorObs(localVelocity.x);
            AddVectorObs(localVelocity.z);
            AddVectorObs(System.Convert.ToInt32(frozen));
            AddVectorObs(System.Convert.ToInt32(shoot));
        }
    }

    public Color32 ToColor(int hexVal)
    {
        byte r = (byte)((hexVal >> 16) & 0xFF);
        byte g = (byte)((hexVal >> 8) & 0xFF);
        byte b = (byte)(hexVal & 0xFF);
        return new Color32(r, g, b, 255);
    }

    public void MoveAgent(float[] act)
    {
        shoot = false;

        if (Time.time > frozenTime + 4f && frozen)
        {
            Unfreeze();
        }
        if (Time.time > effectTime + 0.5f)
        {
            if (poisioned)
            {
                Unpoison();
            }
            if (satiated)
            {
                Unsatiate();
            }
        }

        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        if (!frozen)
        {
            bool shootCommand = false;
            if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
                rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
                shootCommand = Mathf.Clamp(act[2], -1f, 1f) > 0.5f;
            }
            else
            {
                var forwardAxis = (int)act[0];
                var rightAxis = (int)act[1];
                var rotateAxis = (int)act[2];
                var shootAxis = (int)act[3];
                
                switch (forwardAxis)
                {
                    case 1:
                        dirToGo = transform.forward;
                        break;
                    case 2:
                        dirToGo = -transform.forward;
                        break;
                }
                
                switch (rightAxis)
                {
                    case 1:
                        dirToGo = transform.right;
                        break;
                    case 2:
                        dirToGo = -transform.right;
                        break;
                }

                switch (rotateAxis)
                {
                    case 1:
                        rotateDir = -transform.up;
                        break;
                    case 2:
                        rotateDir = transform.up;
                        break; 
                }
                switch (shootAxis)
                {
                    case 1:
                        shootCommand = true;
                        break;
                }
            }
            if (shootCommand)
            {
                shoot = true;
                dirToGo *= 0.5f;
                agentRb.velocity *= 0.75f;
            }
            agentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
            transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);
        }

        if (agentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            agentRb.velocity *= 0.95f;
        }

        if (shoot)
        {
            myLaser.transform.localScale = new Vector3(1f, 1f, 1f);
            Vector3 position = transform.TransformDirection(RayPerception.PolarToCartesian(25f, 90f));
            Debug.DrawRay(transform.position, position, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 2f, position, out hit, 25f))
            {
                if (hit.collider.gameObject.CompareTag("agent"))
                {
                    hit.collider.gameObject.GetComponent<BananaAgent>().Freeze();
                }
            }
        }
        else
        {
            myLaser.transform.localScale = new Vector3(0f, 0f, 0f);

        }
    }


    void Freeze()
    {
        gameObject.tag = "frozenAgent";
        frozen = true;
        frozenTime = Time.time;
        gameObject.GetComponent<Renderer>().material = frozenMaterial;
    }


    void Unfreeze()
    {
        frozen = false;
        gameObject.tag = "agent";
        gameObject.GetComponent<Renderer>().material = normalMaterial;
    }

    void Poison()
    {
        poisioned = true;
        effectTime = Time.time;
        gameObject.GetComponent<Renderer>().material = badMaterial;
    }

    void Unpoison()
    {
        poisioned = false;
        gameObject.GetComponent<Renderer>().material = normalMaterial;
    }

    void Satiate()
    {
        satiated = true;
        effectTime = Time.time;
        gameObject.GetComponent<Renderer>().material = goodMaterial;
    }

    void Unsatiate()
    {
        satiated = false;
        gameObject.GetComponent<Renderer>().material = normalMaterial;
    }



    public override void AgentAction(float[] vectorAction, string textAction)
    {
        MoveAgent(vectorAction);
    }

    public override void AgentReset()
    {
        Unfreeze();
        Unpoison();
        Unsatiate();
        shoot = false;
        agentRb.velocity = Vector3.zero;
        bananas = 0;
        myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        transform.position = new Vector3(Random.Range(-myArea.range, myArea.range),
                                         2f, Random.Range(-myArea.range, myArea.range))
            + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("banana"))
        {
            Satiate();
            collision.gameObject.GetComponent<BananaLogic>().OnEaten();
            AddReward(1f);
            bananas += 1;
            if (contribute)
            {
                myAcademy.totalScore += 1;
            }
        }
        if (collision.gameObject.CompareTag("badBanana"))
        {
            Poison();
            collision.gameObject.GetComponent<BananaLogic>().OnEaten();

            AddReward(-1f);
            if (contribute)
            {
                myAcademy.totalScore -= 1;
            }
        }
    }

    public override void AgentOnDone()
    {

    }
}
