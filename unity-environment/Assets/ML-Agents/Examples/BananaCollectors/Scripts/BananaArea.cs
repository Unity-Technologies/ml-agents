using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class BananaArea : Area
{
    public GameObject banana;
    public GameObject badBanana;
    public int numBananas;
    public int numBadBananas;
    public bool respawnBananas;
    public float range;
    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void CreateBanana(int numBana, GameObject bananaType)
    {
        for (int i = 0; i < numBana; i++)
        {
            GameObject bana = Instantiate(bananaType, new Vector3(Random.Range(-range, range), 1f,
                                                              Random.Range(-range, range)) + transform.position,
                                          Quaternion.Euler(new Vector3(0f, Random.Range(0f, 360f), 90f)));
            bana.GetComponent<BananaLogic>().respawn = respawnBananas;
            bana.GetComponent<BananaLogic>().myArea = this;
        }
    }

    public void ResetBananaArea(GameObject[] agents)
    {
        foreach (GameObject agent in agents)
        {
            if (agent.transform.parent == gameObject.transform)
            {
                agent.transform.position = new Vector3(Random.Range(-range, range), 2f,
                                                       Random.Range(-range, range))
                    + transform.position;
                agent.transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
            }
        }

        CreateBanana(numBananas, banana);
        CreateBanana(numBadBananas, badBanana);
    }

    public override void ResetArea()
    {
    }
}
