using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PyramidAgent : Agent
{
    public GameObject myAcademyObj;
    public GameObject area;
    PyramidArea myArea;
    Rigidbody agentRB;
    public float turnSpeed;
    public float xForce;
    public float yForce;
    public float zForce;
    public bool contribute;

    public override void InitializeAgent()
    {
        base.InitializeAgent();
        agentRB = GetComponent<Rigidbody>(); //cache the RB
        Monitor.verticalOffset = 1f;
        myArea = area.GetComponent<PyramidArea>();
    }

    public override void CollectObservations()
    {
        float rayDistance = 50f;
        float[] rayAngles = { 20f, 90f, 160f, 45f, 135f, 70f, 110f };
        float[] rayAngles1 = { 25f, 95f, 165f, 50f, 140f, 75f, 115f };
        float[] rayAngles2 = { 15f, 85f, 155f, 40f, 130f, 65f, 105f };

        string[] detectableObjects = { "block", "wall", "goal"};
        RayPerception(rayDistance, rayAngles, detectableObjects, 0f);
        RayPerception(rayDistance, rayAngles1, detectableObjects, 5f);
        RayPerception(rayDistance, rayAngles2, detectableObjects, 10f);
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
            if (Physics.SphereCast(transform.position, .1f, position, out hit, rayDistance))
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

    public void MoveAgent(float[] act)
    {
        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        //If we're using Continuous control you will need to change the Action
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
        transform.Rotate(rotateDir, Time.deltaTime * 200f);
        agentRB.AddForce(dirToGo * 2f, ForceMode.VelocityChange); //GO
    }

    public override void AgentAction(float[] act)
    {
        MoveAgent(act);
    }

    public override void AgentReset()
    {
        agentRB.velocity = Vector3.zero;
        transform.position = new Vector3(Random.Range(-myArea.range, myArea.range), 
                                         2f, Random.Range(-myArea.range, myArea.range)) 
            + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("goal"))
        {
            AddReward(1f);
            Destroy(collision.gameObject.transform.parent.gameObject);
            area.GetComponent<PyramidArea>().CreateObject(1);
        }
    }

    public override void AgentOnDone()
    {

    }
}
