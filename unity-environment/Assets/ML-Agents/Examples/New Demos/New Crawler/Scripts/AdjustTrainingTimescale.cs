using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AdjustTrainingTimescale : MonoBehaviour {
    // float oldtimeScale;
	
	// Update is called once per frame
	void Update ()
    {
        
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Time.timeScale = 1f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            Time.timeScale = 2f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            Time.timeScale = 3f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            Time.timeScale = 4f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            Time.timeScale = 5f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            Time.timeScale = 6f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            Time.timeScale = 7f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha8))
        {
            Time.timeScale = 8f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            Time.timeScale = 9f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            Time.timeScale *= 2f;
        }




        // if (Input.GetKeyDown(KeyCode.Alpha0))
        // {
        //     Time.timeScale =1f;
        // }
        // if (Input.GetKeyDown(KeyCode.Space))
        // {
        //     if (Time.timeScale != 1f)
        //     {
        //         oldtimeScale = Time.timeScale;
        //         Time.timeScale = 1f;
        //     }
        //     else
        //     {
        //         Time.timeScale = oldtimeScale;
        //     }
        // }
        
	}
}
