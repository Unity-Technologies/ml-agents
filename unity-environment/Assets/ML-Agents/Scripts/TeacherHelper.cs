using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TeacherHelper : MonoBehaviour {

    bool recordExperiences;
    bool resetBuffer;
    Agent myAgent;
    float bufferResetTime;

	// Use this for initialization
	void Start () {
        recordExperiences = true;
        resetBuffer = false;
        myAgent = GetComponent<Agent>();
        bufferResetTime = Time.time;
	}
	
	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(KeyCode.R))
        {
            recordExperiences = !recordExperiences;
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            resetBuffer = true;
            bufferResetTime = Time.time;
        }
        else
        {
            resetBuffer = false;
        }
        Monitor.Log("Recording experiences", recordExperiences.ToString());
        float timeSinceBufferReset = Time.time - bufferResetTime;
        Monitor.Log("Seconds since buffer reset", Mathf.FloorToInt(timeSinceBufferReset));
	}

    void FixedUpdate()
    {
        // Convert both bools into single comma separated string. Python makes
        // assumption that this structure is preserved. 
        myAgent.SetTextObs(recordExperiences.ToString() + "," + resetBuffer.ToString());
    }
}
