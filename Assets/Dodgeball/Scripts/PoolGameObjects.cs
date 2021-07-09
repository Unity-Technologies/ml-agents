using System.Collections.Generic;
using UnityEngine;

public class PoolGameObjects : MonoBehaviour
{
    public int numberOfObjectsToPool;
    public GameObject objectPrefab;
    public bool setupComplete;
    public List<GameObject> poolList;
    public List<Rigidbody> rbList;

    void Awake()
    {
        PoolObjects();
    }

    public void PoolObjects()
    {
        poolList = new List<GameObject>(numberOfObjectsToPool);
        for (int i = 0; i < numberOfObjectsToPool; i++)
        {
            GameObject obj = Instantiate(objectPrefab, transform.position,
                Quaternion.identity); //will be parent of rhythmNodes
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            if (rb)
            {
                rbList.Add(rb);
            }

            obj.transform.SetParent(transform); //parent the GameObject
            obj.SetActive(false);
            poolList.Add(obj);
        }

        setupComplete = true;
    }
}
