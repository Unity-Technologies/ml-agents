using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAgentConfigurable : Agent
{

    public float strength;
    float x_position;
    [HideInInspector]
    public bool[] leg_touching;
    [HideInInspector]
    public bool fell;
    Vector3 past_velocity;
    Transform body;
    Rigidbody bodyRB;
    public Transform[] limbs;
    Rigidbody[] limbRBs;
    Dictionary<GameObject, Vector3> transformsPosition;
    Dictionary<GameObject, Quaternion> transformsRotation;

    public override void InitializeAgent()
    {
        body = transform.Find("Sphere");
        bodyRB = body.GetComponent<Rigidbody>();
        transformsPosition = new Dictionary<GameObject, Vector3>();
        transformsRotation = new Dictionary<GameObject, Quaternion>();
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            transformsPosition[child.gameObject] = child.position;
            transformsRotation[child.gameObject] = child.rotation;
        }
        leg_touching = new bool[4];
        limbRBs = new Rigidbody[limbs.Length];
        for (int i = 0; i < limbs.Length; i++)
        {
            limbRBs[i] = limbs[i].gameObject.GetComponent<Rigidbody>();
        }
    }

    public override void CollectObservations()
    {
        AddVectorObs(body.transform.rotation.eulerAngles);

        AddVectorObs(bodyRB.velocity);

        AddVectorObs((bodyRB.velocity - past_velocity) / Time.fixedDeltaTime);
        past_velocity = bodyRB.velocity;

        for (int i = 0; i < limbs.Length; i++)
        {
            AddVectorObs(limbs[i].localPosition);
            AddVectorObs(limbs[i].localRotation);
            AddVectorObs(limbRBs[i].velocity);
            AddVectorObs(limbRBs[i].angularVelocity);
        }

        for (int index = 0; index < 4; index++)
        {
            if (leg_touching[index])
            {
                AddVectorObs(1);
            }
            else
            {
                AddVectorObs(0);
            }
            leg_touching[index] = false;
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        for (int k = 0; k < vectorAction.Length; k++)
        {
            vectorAction[k] = Mathf.Clamp(vectorAction[k], -1f, 1f);
        }

        limbRBs[0].AddTorque(-limbs[0].transform.right * strength * vectorAction[0]);
        limbRBs[1].AddTorque(-limbs[1].transform.right * strength * vectorAction[1]);
        limbRBs[2].AddTorque(-limbs[2].transform.right * strength * vectorAction[2]);
        limbRBs[3].AddTorque(-limbs[3].transform.right * strength * vectorAction[3]);
        limbRBs[0].AddTorque(-body.transform.up * strength * vectorAction[4]);
        limbRBs[1].AddTorque(-body.transform.up * strength * vectorAction[5]);
        limbRBs[2].AddTorque(-body.transform.up * strength * vectorAction[6]);
        limbRBs[3].AddTorque(-body.transform.up * strength * vectorAction[7]);
        limbRBs[4].AddTorque(-limbs[4].transform.right * strength * vectorAction[8]);
        limbRBs[5].AddTorque(-limbs[5].transform.right * strength * vectorAction[9]);
        limbRBs[6].AddTorque(-limbs[6].transform.right * strength * vectorAction[10]);
        limbRBs[7].AddTorque(-limbs[7].transform.right * strength * vectorAction[11]);

        float torque_penalty = vectorAction[0] * vectorAction[0] + 
            vectorAction[1] * vectorAction[1] + 
            vectorAction[2] * vectorAction[2] + 
            vectorAction[3] * vectorAction[3] +
            vectorAction[4] * vectorAction[4] + 
            vectorAction[5] * vectorAction[5] + 
            vectorAction[6] * vectorAction[6] + 
            vectorAction[7] * vectorAction[7] + 
            vectorAction[8] * vectorAction[8] + 
            vectorAction[9] * vectorAction[9] + 
            vectorAction[10] * vectorAction[10] + 
            vectorAction[11] * vectorAction[11];

        if (!IsDone())
        {
            SetReward(0 - 0.01f * torque_penalty + 1.0f * bodyRB.velocity.x
            - 0.05f * Mathf.Abs(body.transform.position.z - body.transform.parent.transform.position.z)
                      - 0.05f * Mathf.Abs(bodyRB.velocity.y)
            );
        }
        if (fell)
        {
            Done();
            AddReward(-1f);
        }
    }

    public override void AgentReset()
    {
        fell = false;
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if ((child.gameObject.name.Contains("Crawler"))
                || (child.gameObject.name.Contains("parent")))
            {
                continue;
            }
            child.position = transformsPosition[child.gameObject];
            child.rotation = transformsRotation[child.gameObject];
            child.gameObject.GetComponent<Rigidbody>().velocity = default(Vector3);
            child.gameObject.GetComponent<Rigidbody>().angularVelocity = default(Vector3);
        }
        gameObject.transform.rotation = Quaternion.Euler(new Vector3(0, Random.value * 90 - 45, 0));
    }

    public override void AgentOnDone()
    {

    }
}
