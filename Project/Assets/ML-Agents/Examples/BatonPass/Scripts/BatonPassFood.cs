using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BatonPassFood : MonoBehaviour
{

    public BatonPassArea Area;

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("agent"))
        {
            this.gameObject.SetActive(false);
            Area.FoodEaten();

        }
    }
}
