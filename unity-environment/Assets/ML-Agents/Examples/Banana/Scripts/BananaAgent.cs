using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaAgent : Agent
{
    public GameObject area;
    bool frozen;
    float frozenTime;
    Rigidbody agentRB;
    public float turnSpeed;
    public float xForce;
    public float yForce;
    public float zForce;
    public Material normalMaterial;
    int bananas;
    public GameObject myLazer;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        agentRB = GetComponent<Rigidbody>(); //cache the RB
        Monitor.verticalOffset = 1f;

    }

    public override List<float> CollectState()
    {
        float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(GiveCatersian(25f, noisyAngle));
            Debug.DrawRay(transform.position, position, Color.green, 0.0f, true);
            RaycastHit hit;
            float[] subList = { 0f, 0f, 0f, 0f, 0f, 0f };
            if (Physics.SphereCast(transform.position, 1.0f, position, out hit, 25f))
            {
                if (hit.collider.gameObject.CompareTag("banana"))
                {
                    subList[1] = 1f;
                }
                if (hit.collider.gameObject.CompareTag("agent"))
                {
                    subList[2] = 1f;
                }
                if (hit.collider.gameObject.CompareTag("wall"))
                {
                    subList[3] = 1f;
                }
                if (hit.collider.gameObject.CompareTag("badBanana"))
                {
                    subList[4] = 1f;
                }
                subList[5] = hit.distance / 25f;
            }
            else
            {
                subList[0] = 1f;
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

    public float DegreeToRadian(float degree){
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
        Monitor.Log("Bananas", bananas, MonitorType.text, gameObject.transform);

        if (Time.time > frozenTime + 4f) {
            frozen = false;
            gameObject.GetComponent<Renderer>().material = normalMaterial;
        }


        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        //int movement = Mathf.FloorToInt(act[0]);
        bool shoot = false;


        if (!frozen)
        {
            dirToGo = transform.forward * Mathf.Clamp(act[0], 0f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
            if (Mathf.Clamp(act[2], 0f, 1f) > 0.5f) { 
                shoot = true; 
            }
            else {
                agentRB.AddForce(new Vector3(dirToGo.x * xForce, dirToGo.y * yForce, dirToGo.z * zForce), ForceMode.Acceleration);
                transform.Rotate(rotateDir, Time.deltaTime * turnSpeed);
            }
        }

        if (agentRB.velocity.sqrMagnitude > 25f) //slow it down
        {
            agentRB.velocity *= 0.95f;
        }

        if (shoot) {
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
        else {
            myLazer.transform.localScale = new Vector3(0f, 0f, 0f);

        }

    }

    public void Freeze() {
        frozen = true;
        frozenTime = Time.time;
        gameObject.GetComponent<Renderer>().material.color = Color.black;
    }

    void RotateTowards(Vector3 pos)
    {
        Vector3 dirToBox = pos - transform.position; //get dir
        dirToBox.y = 0;
        Quaternion targetRotation = Quaternion.LookRotation(dirToBox); //get our needed rotation
        agentRB.MoveRotation(Quaternion.Lerp(agentRB.transform.rotation, targetRotation, Time.deltaTime * turnSpeed));
    }

    public override void AgentStep(float[] act)
    {
        MoveAgent(act);
    }

    public override void AgentReset()
    {
        agentRB.velocity = Vector3.zero;
        frozen = false;
        bananas = 0;
        myLazer.transform.localScale = new Vector3(0f, 0f, 0f);
        gameObject.GetComponent<Renderer>().material = normalMaterial;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("banana"))
        {
            collision.gameObject.GetComponent<BananaLogic>().OnEaten();
            reward += 1f;
            bananas += 1;
        }
        if (collision.gameObject.CompareTag("badBanana"))
        {
            collision.gameObject.GetComponent<BananaLogic>().OnEaten();
            reward -= 1f;
        }
    }


    public override void AgentOnDone()
    {

    }
}
