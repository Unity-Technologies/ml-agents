using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Behavioral Cloning Helper script. Attach to teacher agent to enable 
/// resetting the experience buffer, as well as toggling session recording.
/// </summary>
public class BCTeacherHelper : MonoBehaviour {

    bool recordExperiences;
    bool resetBuffer;
    Agent myAgent;
    float bufferResetTime;

    public KeyCode recordKey = KeyCode.R;
    public KeyCode resetKey = KeyCode.C;

    // Use this for initialization
    void Start () {
        recordExperiences = true;
        resetBuffer = false;
        myAgent = GetComponent<Agent>();
        bufferResetTime = Time.time;
    }
    
    // Update is called once per frame
    void Update () {
        if (Input.GetKeyDown(recordKey))
        {
            recordExperiences = !recordExperiences;
        }
        if (Input.GetKeyDown(resetKey))
        {
            resetBuffer = true;
            bufferResetTime = Time.time;
        }
        else
        {
            resetBuffer = false;
        }
        Monitor.Log("Recording experiences " + recordKey.ToString(), recordExperiences.ToString());
        float timeSinceBufferReset = Time.time - bufferResetTime;
        Monitor.Log("Seconds since buffer reset " + resetKey.ToString(), Mathf.FloorToInt(timeSinceBufferReset));
    }

    void FixedUpdate()
    {
        // Convert both bools into single comma separated string. Python makes
        // assumption that this structure is preserved. 
        myAgent.SetTextObs(recordExperiences.ToString() + "," + resetBuffer.ToString());
    }
}
