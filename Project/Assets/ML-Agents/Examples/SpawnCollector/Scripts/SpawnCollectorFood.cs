using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnCollectorFood : MonoBehaviour
{

    public SpawnArea Area;

    void Start()
    {
        Area.RegisterFood(this);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("agent"))
        {
            this.gameObject.SetActive(false);
            Area.FoodEaten();

        }
    }
}
