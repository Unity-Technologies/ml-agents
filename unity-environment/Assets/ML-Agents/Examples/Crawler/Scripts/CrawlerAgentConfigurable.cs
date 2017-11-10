using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerAgentConfigurable: Agent
{

    public float strength;

    float x_position;


    [HideInInspector]
    public bool[] leg_touching;

    [HideInInspector]
    public bool fell;

    Vector3 past_velocity;

    Transform body;


    public Transform[] limbs;



    //
    Dictionary<GameObject, Vector3> transformsPosition;
    Dictionary<GameObject, Quaternion> transformsRotation;




    public override void InitializeAgent()
    {
        
        body = transform.Find("Sphere");
        


        transformsPosition = new Dictionary<GameObject, Vector3>();
        transformsRotation = new Dictionary<GameObject, Quaternion>();
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            transformsPosition[child.gameObject] = child.position;
            transformsRotation[child.gameObject] = child.rotation;
        }



        leg_touching = new bool[4];

    }

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        state.Add(body.transform.rotation.eulerAngles.x);
        state.Add(body.transform.rotation.eulerAngles.y);
        state.Add(body.transform.rotation.eulerAngles.z);

        state.Add(body.gameObject.GetComponent<Rigidbody>().velocity.x);
        state.Add(body.gameObject.GetComponent<Rigidbody>().velocity.y);
        state.Add(body.gameObject.GetComponent<Rigidbody>().velocity.z);

        state.Add((body.gameObject.GetComponent<Rigidbody>().velocity.x - past_velocity.x) / Time.fixedDeltaTime);
        state.Add((body.gameObject.GetComponent<Rigidbody>().velocity.y - past_velocity.y) / Time.fixedDeltaTime);
        state.Add((body.gameObject.GetComponent<Rigidbody>().velocity.z - past_velocity.z) / Time.fixedDeltaTime);
        past_velocity = body.gameObject.GetComponent<Rigidbody>().velocity;

        foreach (Transform t in limbs)
        {
            state.Add(t.localPosition.x);
            state.Add(t.localPosition.y);
            state.Add(t.localPosition.z);
            state.Add(t.localRotation.x);
            state.Add(t.localRotation.y);
            state.Add(t.localRotation.z);
            state.Add(t.localRotation.w);
            Rigidbody rb = t.gameObject.GetComponent < Rigidbody >();
            state.Add(rb.velocity.x);
            state.Add(rb.velocity.y);
            state.Add(rb.velocity.z);
            state.Add(rb.angularVelocity.x);
            state.Add(rb.angularVelocity.y);
            state.Add(rb.angularVelocity.z);
        }




        for (int index = 0; index < 4; index++)
        {
            if (leg_touching[index])
            {
                state.Add(1.0f);
            }
            else
            {
                state.Add(0.0f);
            }
            leg_touching[index] = false;
        }






        return state;
    }

    public override void AgentStep(float[] act)
    {
        for (int k = 0; k < act.Length; k++)
        {
            act[k] = Mathf.Max(Mathf.Min(act[k], 1), -1);
        }

        limbs[0].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[0].transform.right * strength * act[0]);
        limbs[1].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[1].transform.right * strength * act[1]);
        limbs[2].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[2].transform.right * strength * act[2]);
        limbs[3].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[3].transform.right * strength * act[3]);

        limbs[0].gameObject.GetComponent<Rigidbody>().AddTorque(-body.transform.up * strength * act[4]);
        limbs[1].gameObject.GetComponent<Rigidbody>().AddTorque(-body.transform.up * strength * act[5]);
        limbs[2].gameObject.GetComponent<Rigidbody>().AddTorque(-body.transform.up * strength * act[6]);
        limbs[3].gameObject.GetComponent<Rigidbody>().AddTorque(-body.transform.up * strength * act[7]);


        limbs[4].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[4].transform.right * strength * act[8]);
        limbs[5].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[5].transform.right * strength * act[9]);
        limbs[6].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[6].transform.right * strength * act[10]);
        limbs[7].gameObject.GetComponent<Rigidbody>().AddTorque(-limbs[7].transform.right * strength * act[11]);



        float torque_penalty = act[0] * act[0] + act[1] * act[1] + act[2] * act[2] + act[3] * act[3]
                         + act[4] * act[4] + act[5] * act[5] + act[6] * act[6] + act[7] * act[7]
                         + act[8] * act[8] + act[9] * act[9] + act[10] * act[10] + act[11] * act[11];

        if (!done)
        {

            reward = (0
            - 0.01f * torque_penalty
            + 1.0f * body.GetComponent<Rigidbody>().velocity.x
            - 0.05f * Mathf.Abs(body.transform.position.z - body.transform.parent.transform.position.z)
            - 0.05f * Mathf.Abs(body.GetComponent<Rigidbody>().velocity.y)
            );
        }
        if (fell)
        {
            done = true;
            reward = -1;
            fell = false;
        }

        Monitor.Log("Reward", reward, MonitorType.slider, body.gameObject.transform);
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {

        }

    }

    public override void AgentReset()
    {

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
