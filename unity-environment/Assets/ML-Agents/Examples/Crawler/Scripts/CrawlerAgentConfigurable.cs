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
        AddVectorObs(body.transform.rotation.eulerAngles.x);
        AddVectorObs(body.transform.rotation.eulerAngles.y);
        AddVectorObs(body.transform.rotation.eulerAngles.z);

        AddVectorObs(bodyRB.velocity.x);
        AddVectorObs(bodyRB.velocity.y);
        AddVectorObs(bodyRB.velocity.z);

        AddVectorObs((bodyRB.velocity.x - past_velocity.x) / Time.fixedDeltaTime);
        AddVectorObs((bodyRB.velocity.y - past_velocity.y) / Time.fixedDeltaTime);
        AddVectorObs((bodyRB.velocity.z - past_velocity.z) / Time.fixedDeltaTime);
        past_velocity = bodyRB.velocity;

        for (int i = 0; i < limbs.Length; i++)
        {
            AddVectorObs(limbs[i].localPosition.x);
            AddVectorObs(limbs[i].localPosition.y);
            AddVectorObs(limbs[i].localPosition.z);
            AddVectorObs(limbs[i].localRotation.x);
            AddVectorObs(limbs[i].localRotation.y);
            AddVectorObs(limbs[i].localRotation.z);
            AddVectorObs(limbs[i].localRotation.w);
            AddVectorObs(limbRBs[i].velocity.x);
            AddVectorObs(limbRBs[i].velocity.y);
            AddVectorObs(limbRBs[i].velocity.z);
            AddVectorObs(limbRBs[i].angularVelocity.x);
            AddVectorObs(limbRBs[i].angularVelocity.y);
            AddVectorObs(limbRBs[i].angularVelocity.z);
        }

        for (int index = 0; index < 4; index++)
        {
            if (leg_touching[index])
            {
                AddVectorObs(1.0f);
            }
            else
            {
                AddVectorObs(0.0f);
            }
            leg_touching[index] = false;
        }
    }

    public override void AgentAction(float[] act)
    {
        for (int k = 0; k < act.Length; k++)
        {
            act[k] = Mathf.Clamp(act[k], -1f, 1f);
        }

        limbRBs[0].AddTorque(-limbs[0].transform.right * strength * act[0]);
        limbRBs[1].AddTorque(-limbs[1].transform.right * strength * act[1]);
        limbRBs[2].AddTorque(-limbs[2].transform.right * strength * act[2]);
        limbRBs[3].AddTorque(-limbs[3].transform.right * strength * act[3]);
        limbRBs[0].AddTorque(-body.transform.up * strength * act[4]);
        limbRBs[1].AddTorque(-body.transform.up * strength * act[5]);
        limbRBs[2].AddTorque(-body.transform.up * strength * act[6]);
        limbRBs[3].AddTorque(-body.transform.up * strength * act[7]);
        limbRBs[4].AddTorque(-limbs[4].transform.right * strength * act[8]);
        limbRBs[5].AddTorque(-limbs[5].transform.right * strength * act[9]);
        limbRBs[6].AddTorque(-limbs[6].transform.right * strength * act[10]);
        limbRBs[7].AddTorque(-limbs[7].transform.right * strength * act[11]);

        float torque_penalty = act[0] * act[0] + act[1] * act[1] + act[2] * act[2] + act[3] * act[3]
                         + act[4] * act[4] + act[5] * act[5] + act[6] * act[6] + act[7] * act[7]
                         + act[8] * act[8] + act[9] * act[9] + act[10] * act[10] + act[11] * act[11];

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
