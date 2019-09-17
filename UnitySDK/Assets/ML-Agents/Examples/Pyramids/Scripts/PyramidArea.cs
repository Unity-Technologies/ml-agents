using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class PyramidArea : Area
{
    public GameObject pyramid;
    public GameObject stonePyramid;
    public GameObject[] spawnAreas;
    public int numPyra;
    public float range;

    public void CreatePyramid(int numObjects, int spawnAreaIndex)
    {
        CreateObject(numObjects, pyramid, spawnAreaIndex);
    }
    
    public void CreateStonePyramid(int numObjects, int spawnAreaIndex)
    {
        CreateObject(numObjects, stonePyramid, spawnAreaIndex);
    }
    
    private void CreateObject(int numObjects, GameObject desiredObject, int spawnAreaIndex)
    {
        for (var i = 0; i < numObjects; i++)
        {
            var newObject = Instantiate(desiredObject, Vector3.zero, 
                Quaternion.Euler(0f, 0f, 0f), transform);
            PlaceObject(newObject, spawnAreaIndex);
        }
    }

    public void PlaceObject(GameObject objectToPlace, int spawnAreaIndex)
    {
        var spawnTransform = spawnAreas[spawnAreaIndex].transform;
        var xRange = spawnTransform.localScale.x / 2.1f;
        var zRange = spawnTransform.localScale.z / 2.1f;
        
        objectToPlace.transform.position = new Vector3(Random.Range(-xRange, xRange), 2f, Random.Range(-zRange, zRange)) 
                                            + spawnTransform.position;
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
