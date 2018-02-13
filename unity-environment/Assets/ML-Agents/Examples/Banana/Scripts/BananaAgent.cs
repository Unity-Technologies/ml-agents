using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaAgent : Agent
{
    public GameObject myAcademyObj;
    BananaAcademy myAcademy;
    public GameObject area;
    BananaArea myArea;
    bool frozen;
    bool poisioned;
    bool satiated;
    bool shoot;
    float frozenTime;
    float effectTime;
    Rigidbody agentRB;
    public float turnSpeed;
    public float xForce;
    public float yForce;
    public float zForce;
    public Material normalMaterial;
    public Material badMaterial;
    public Material goodMaterial;
    int bananas;
    public GameObject myLazer;
    public bool contribute;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        agentRB = GetComponent<Rigidbody>(); //cache the RB
        Monitor.verticalOffset = 1f;
        myArea = area.GetComponent<BananaArea>();
        myAcademy = myAcademyObj.GetComponent<BananaAcademy>();
    }

    public override List<float> CollectState()
    {
        float rayDistance = 50f;
        float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
        string[] detectableObjects = { "banana", "agent", "wall", "badBanana", "frozenAgent" };
        state = RayPerception(state, rayDistance, rayAngles, detectableObjects);
        Vector3 localVelocity = transform.InverseTransformDirection(agentRB.velocity);
        state.Add(localVelocity.x);
        state.Add(localVelocity.z);
        state.Add(System.Convert.ToInt32(frozen));
        state.Add(System.Convert.ToInt32(shoot));
        return state;
    }

    public List<float> RayPerception(List<float> state, float rayDistance, 
                                     float[] rayAngles, string[] detectableObjects)
    {
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(GiveCatersian(rayDistance, noisyAngle));
            Debug.DrawRay(transform.position, position, Color.green, 0.0f, true);
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
            state.AddRange(new List<float>(subList));
        }
        return state;
    }

    public Vector3 GiveCatersian(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian(angle));
        float z = radius * Mathf.Sin(DegreeToRadian(angle));
        return new Vector3(x, -0.1f, z);
    }

    public float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }

    public Color32 ToColor(int HexVal)
    {
        byte R = (byte)((HexVal >> 16) & 0xFF);
        byte G = (byte)((HexVal >> 8) & 0xFF);
        byte B = (byte)((HexVal) & 0xFF);
        return new Color32(R, G, B, 255);
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
            dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
            if (Mathf.Clamp(act[2], 0f, 1f) > 0.5f)
            {
                shoot = true;
                dirToGo *= 0.5f;
                agentRB.velocity *= 0.75f;
            }
            agentRB.AddForce(new Vector3(dirToGo.x * xForce, dirToGo.y * yForce, dirToGo.z * zForce), 
                             ForceMode.Acceleration);
            transform.Rotate(rotateDir, Time.deltaTime * turnSpeed);
        }

        if (agentRB.velocity.sqrMagnitude > 25f) //slow it down
        {
            agentRB.velocity *= 0.95f;
        }

        if (shoot)
        {
            myLazer.transform.localScale = new Vector3(1f, 1f, 1f);
            Vector3 position = transform.TransformDirection(GiveCatersian(25f, 90f));
            Debug.DrawRay(transform.position, position, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 2f, position, out hit, 25f))
            {
                if (hit.collider.gameObject.tag == "agent")
                {
                    hit.collider.gameObject.GetComponent<BananaAgent>().Freeze();
                }
            }
        }
        else
        {
            myLazer.transform.localScale = new Vector3(0f, 0f, 0f);

        }

    }

    void Freeze()
    {
        gameObject.tag = "frozenAgent";
        frozen = true;
        frozenTime = Time.time;
        gameObject.GetComponent<Renderer>().material.color = Color.black;
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



    public override void AgentStep(float[] act)
    {
        MoveAgent(act);
    }

    public override void AgentReset()
    {
        Unfreeze();
        Unpoison();
        Unsatiate();
        shoot = false;
        agentRB.velocity = Vector3.zero;
        bananas = 0;
        myLazer.transform.localScale = new Vector3(0f, 0f, 0f);
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
            reward += 1f;
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
            reward -= 1f;
            if (contribute)
            {
                myAcademy.totalScore -= 1;
            }
        }
        if (collision.gameObject.CompareTag("wall"))
        {
            done = true;
        }
    }

    public override void AgentOnDone()
    {

    }
}
