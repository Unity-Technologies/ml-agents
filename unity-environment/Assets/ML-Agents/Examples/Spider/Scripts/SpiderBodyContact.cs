using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpiderBodyContact : MonoBehaviour {

    SpiderAgentConfigurable agent;

    void Start(){
        agent = gameObject.transform.parent.gameObject.GetComponent<SpiderAgentConfigurable>();
    }

    void OnTriggerEnter(Collider other){
        if (other.gameObject.name == "Ground")
        {
            agent.fell = true;
        }
    }
}
