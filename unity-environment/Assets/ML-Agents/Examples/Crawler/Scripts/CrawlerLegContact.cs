using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrawlerLegContact : MonoBehaviour {

    public int index;
    public CrawlerAgentConfigurable agent;

    void Start(){
//        agent = gameObject.transform.parent.gameObject.GetComponent<SpiderAgent>();
    }

    void OnCollisionStay(Collision other){
        if (other.gameObject.name == "Platform")
        {
            agent.leg_touching[index] = true;
        }
    }

}
