using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaArea : Area
{

    BananaAcademy aca;
    public GameObject banana;
    public GameObject badBanana;
    public int numBananas;
    public int numBadBananas;
    public bool respawnBananas;

    // Use this for initialization
    void Start()
    {
        aca = GameObject.Find("Academy").GetComponent<BananaAcademy>();
        aca.listArea.Add(this);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public override void ResetArea()
    {
        float range = 45f;

        foreach (GameObject agent in aca.agents)
        {
            if (agent.transform.parent == gameObject.transform)
            {
                agent.transform.position = new Vector3(Random.Range(-range, range), 2f, Random.Range(-range, range)) + transform.position;
                agent.transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
            }
        }

        for (int i = 0; i < numBananas; i++)
        {
            GameObject bana = Instantiate(banana, new Vector3(Random.Range(-range, range), 1f, Random.Range(-range, range)) + transform.position, banana.gameObject.transform.rotation);
            bana.GetComponent<BananaLogic>().respawn = respawnBananas;
        }
        for (int i = 0; i < numBadBananas; i++)
        {
            GameObject bana = Instantiate(badBanana, new Vector3(Random.Range(-range, range), 1f, Random.Range(-range, range)) + transform.position, banana.gameObject.transform.rotation);
            bana.GetComponent<BananaLogic>().respawn = respawnBananas;
        }
    }

}
