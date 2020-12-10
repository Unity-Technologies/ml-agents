using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnInCircle : MonoBehaviour
{
    public GameObject Prefab;
    public float Radius = 1000;
    public float Amount = 10000;
    public bool DoIt;

    void Update()
    {
        if (DoIt && Prefab != null)
        {
            var rootPos = transform.position;
            var rot = transform.rotation;
            for (int i = 0; i < Amount; ++i)
            {
                var a = Random.Range(0, 360);
                var pos = new Vector3(Mathf.Cos(a), 0, Mathf.Sin(a));
                pos = rootPos + pos * Mathf.Sqrt(Random.Range(0.0f, 1.0f)) * Radius;
                Instantiate(Prefab, pos, rot, transform.parent);
            }
        }
        DoIt = false;
    }
}
