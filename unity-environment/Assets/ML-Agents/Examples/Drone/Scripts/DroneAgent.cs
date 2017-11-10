using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneAgent : Agent {
    [Header("Specific to Drone")]
    Transform body;
    Rigidbody rb;
    public Transform target;

    public DroneEngine[] engines;
    public float maxPower = 30; 
    public DroneAcademy aca;


    float[] pastValues;
    Dictionary<GameObject, Vector3> transformsPosition;
    Dictionary<GameObject, Quaternion> transformsRotation;

    public override void InitializeAgent()
    {
        body = gameObject.transform.Find("Body");
        rb = body.gameObject.GetComponent<Rigidbody>();
        pastValues = new float[6];
        transformsPosition = new Dictionary<GameObject, Vector3> ();
        transformsRotation = new Dictionary<GameObject, Quaternion> ();
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren) {
            transformsPosition [child.gameObject] = child.position;
            transformsRotation [child.gameObject] = child.rotation;
        }
        foreach (DroneEngine e in engines)
        {
            e.maxPower = maxPower;
        }


    }


    public override List<float> CollectState()
    {
        List<float> state = new List<float>();

        state.Add(body.position.x - target.position.x);
        state.Add(body.position.y - target.position.y);
        state.Add(body.position.z - target.position.z);
//        state.Add(body.rotation.x);
//        state.Add(body.rotation.y);
//        state.Add(body.rotation.z);
//        state.Add(body.rotation.w);
        Vector3 rot = body.rotation.eulerAngles;
        state.Add((rot.x+180)%360-180f);
        state.Add((rot.y+180)%360-180f);
        state.Add((rot.z+180)%360-180f);
        state.Add(body.forward.x);
        state.Add(body.forward.y);
        state.Add(body.forward.z);
        state.Add(body.right.x);
        state.Add(body.right.y);
        state.Add(body.right.z);

        state.Add(rb.velocity.x);
        state.Add(rb.velocity.y);
        state.Add(rb.velocity.z);

        Vector3 angularVel = rb.angularVelocity;
        angularVel.x = ((angularVel.x + 180) % 360 - 180f);
        angularVel.y = ((angularVel.y + 180) % 360 - 180f);
        angularVel.z = ((angularVel.z + 180) % 360 - 180f);
        state.Add(angularVel.x);
        state.Add(angularVel.y);
        state.Add(angularVel.z);

        state.Add((rb.velocity.x - pastValues[0]) / Time.fixedDeltaTime);
        pastValues[0] = rb.velocity.x;
        state.Add((rb.velocity.y - pastValues[1]) / Time.fixedDeltaTime);
        pastValues[1] = rb.velocity.y;
        state.Add((rb.velocity.z - pastValues[2]) / Time.fixedDeltaTime);
        pastValues[2] = rb.velocity.z;

        state.Add((angularVel.x - pastValues[3]) / Time.fixedDeltaTime);
        pastValues[3] = angularVel.x;
        state.Add((angularVel.y - pastValues[4]) / Time.fixedDeltaTime);
        pastValues[4] = angularVel.y;
        state.Add((angularVel.z - pastValues[5]) / Time.fixedDeltaTime);
        pastValues[5] = angularVel.z;

//        state.Add(target.position.x - body.position.x);
//        state.Add(target.position.y - body.position.y);
//        state.Add(target.position.z - body.position.z);

        return state;
    }

    public override void AgentStep(float[] act)
    {
        Monitor.Log("Action", act, MonitorType.hist, body);

        for(int i = 0; i<4 ; i++)
        {
            act[i] = Mathf.Max(-1f, Mathf.Min(act[i], 1f));
        }

        for(int i = 0; i<4 ; i++)
        {
                engines[i].powerMultiplier = act[i];
        }
        if ((target.position - body.position).magnitude > 100f)
        {
            done = true;
            reward = -1f;
        }
        else if ((target.position - body.position).magnitude < aca.resetParameters["targetSize"])
        {
//            done = true;
            reward = 1f;
        }

        else
        {
//            reward = Mathf.Exp(-(target.position - body.position).magnitude / 10f);
            float thrustPenalty = act[0]*act[0] + act[1]*act[1] + act[2]*act[2] +act[3]*act[3];
            reward = (0f
//                + (100f - (target.position - body.position).magnitude) / 100f
                +Mathf.Exp(-(target.position - body.position).magnitude / 10f) / 2f
//                +  Mathf.Max(-1f, Mathf.Min(Vector3.Dot(rb.velocity, (target.position - body.position).normalized)/100f, 1f))
//                - Mathf.Max(rb.velocity.magnitude - 2f, 0f)
//                -0.01f * thrustPenalty
//                +0.01f* Vector3.Dot(body.up, new Vector3(0,1,0))
            );
        }
        Monitor.Log(gameObject.transform.parent.gameObject.name, reward, MonitorType.slider);
        Monitor.Log("Reward", reward, MonitorType.slider, body);

    }

    public override void AgentReset()
    {
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren) {
            if ((child.gameObject.name.Contains("Drone")) 
            )
            {
                continue;
            }
            child.position = transformsPosition [child.gameObject];
            child.rotation = transformsRotation [child.gameObject];
            child.gameObject.GetComponent<Rigidbody> ().velocity = default(Vector3);
            child.gameObject.GetComponent<Rigidbody> ().angularVelocity = default(Vector3);
        }
        target.position = new Vector3(Random.value * 2 - 1, Random.value * 2 - 1, Random.value * 2 - 1) * 20;
        target.localScale = new Vector3(1, 1, 1) * 2 * aca.resetParameters["targetSize"];

    }

    public override void AgentOnDone()
    {

    }
}
