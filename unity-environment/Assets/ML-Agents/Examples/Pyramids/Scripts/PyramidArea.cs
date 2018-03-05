using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PyramidArea : Area
{
    public GameObject mPyra;
    public int numPyra;
    public float range;
    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void CreateObject(int numObjects)
    {
        for (int i = 0; i < numObjects; i++)
        {
            Instantiate(mPyra, new Vector3(Random.Range(-range, range), 1f, Random.Range(-range, range)) + transform.position, 
                                          Quaternion.Euler(0f, 0f, 0f), transform);

        }
    }

    public void ResetPyramidArea(GameObject[] agents)
    {
        foreach (Transform child in transform) if (child.CompareTag("pyramid")) {
                Destroy(child.gameObject);
            }


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

        CreateObject(numPyra);
    }

    public override void ResetArea()
    {
        
    }
}
