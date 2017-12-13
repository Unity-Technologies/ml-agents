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
        List<float> state = new List<float>();

        float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(GiveCatersian(25f, noisyAngle));
            Debug.DrawRay(transform.position, position, Color.green, 0f, true);
            RaycastHit hit;
            float[] subList = { 0f, 0f, 0f, 0f, 0f };
            if (Physics.SphereCast(transform.position, 1.0f, position, out hit, 25f))
            //if (Physics.Raycast(transform.position, position, out hit, 15f))
            {
                if (hit.collider.gameObject.tag == "banana")
                {
                    //print(this.name + " sees a banana at " + noisyAngle.ToString());
                    subList[1] = 1f;
                }
                if (hit.collider.gameObject.tag == "agent")
                {
                    //print(this.name + " sees an agent at " + noisyAngle.ToString());
                    subList[2] = 1f;
                }
                if (hit.collider.gameObject.tag == "wall")
                {
                    //print(this.name + " sees a wall at " + noisyAngle.ToString());
                    subList[3] = 1f;
                }
                subList[4] = hit.distance / 25f;
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

        if (Time.time > frozenTime + 2f) {
            frozen = false;
            gameObject.GetComponent<Renderer>().material = normalMaterial;
        }


        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;
        // Vector3 dirToGoY;
        // Vector3 dirToGoZ;

        int movement = Mathf.FloorToInt(act[0]);
        // print(movement);
        // print(transform.right)
        bool shoot = false;

        if (!frozen)
        {
            if (movement == 1) { rotateDir = -transform.up; } //go left
            if (movement == 2) { rotateDir = transform.up; } //go right
            if (movement == 3) { dirToGo = transform.forward; } //go forward
            if (movement == 4) { dirToGo = -transform.forward; } //go back
            if (movement == 5) { shoot = true; }
            agentRB.AddForce(new Vector3(dirToGo.x * xForce, dirToGo.y * yForce, dirToGo.z * zForce), ForceMode.Acceleration);
            transform.Rotate(rotateDir, Time.deltaTime * turnSpeed);
        }

        // agentRB.AddForce(new Vector3(directionX * 40f, directionY * 300f, directionZ * 40f));
        if (agentRB.velocity.sqrMagnitude > 25f) //slow it down
        {
            agentRB.velocity *= 0.95f;
        }



        if (shoot) {
            myLazer.transform.localScale = new Vector3(1f, 1f, 1f);

            Vector3 position = transform.TransformDirection(GiveCatersian(25f, 90f));
            Debug.DrawRay(transform.position, position, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 1.0f, position, out hit, 25f))
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
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "banana")
        {
            collision.gameObject.GetComponent<BananaLogic>().OnEaten();
            reward += 1f;
            bananas += 1;
        }
    }


    public override void AgentOnDone()
    {

    }
}
