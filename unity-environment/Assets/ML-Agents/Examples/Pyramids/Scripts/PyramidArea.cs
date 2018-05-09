using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PyramidArea : Area
{
    public GameObject pyramid;
    public int numPyra;
    public float range;

    public void CreatePyramid(int numObjects)
    {
        for (var i = 0; i < numObjects; i++)
        {
            Instantiate(pyramid, new Vector3(Random.Range(-range, range), 1f,
                                             Random.Range(-range, range)) + transform.position, 
                        Quaternion.Euler(0f, 0f, 0f), transform);
        }
    }

    public void CleanPyramidArea()
    {
        foreach (Transform child in transform) if (child.CompareTag("pyramid")) 
        {
            Destroy(child.gameObject);
        }
    }

    public override void ResetArea()
    {
        
    }
}
