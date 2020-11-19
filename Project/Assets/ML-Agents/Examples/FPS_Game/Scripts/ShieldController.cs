using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShieldController : MonoBehaviour
{
    public KeyCode shieldKey = KeyCode.I;

    public GameObject shieldGO;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(shieldKey))
        {
            if (!shieldGO.activeInHierarchy)
            {
                shieldGO.SetActive(true);
            }
        }
        else
        {
            if (shieldGO.activeInHierarchy)
            {
                shieldGO.SetActive(false);
            }
        }
    }
}
