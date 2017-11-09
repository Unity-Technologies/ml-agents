using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerBodyContact : MonoBehaviour {

    CrawlerAgentConfigurable agent;

    void Start(){
        agent = gameObject.transform.parent.gameObject.GetComponent<CrawlerAgentConfigurable>();
    }

    void OnTriggerEnter(Collider other){
        if (other.gameObject.name == "Ground")
        {
            agent.fell = true;
        }
    }
}
