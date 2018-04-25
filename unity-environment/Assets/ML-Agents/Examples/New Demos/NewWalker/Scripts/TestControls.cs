using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestControls : MonoBehaviour {

   
    float oldtimeScale;
	
   
	// Update is called once per frame
	void Update ()
    {
        
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Time.timeScale *= 2f;
        }
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            Time.timeScale =1f;
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (Time.timeScale != 1f)
            {
                oldtimeScale = Time.timeScale;
                Time.timeScale = 1f;
            }
            else
            {
                Time.timeScale = oldtimeScale;
            }
        }
        
	}
}
