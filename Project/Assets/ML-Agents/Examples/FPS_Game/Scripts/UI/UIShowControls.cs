using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIShowControls : MonoBehaviour
{
    public GameObject controlsCanvas;

    // Start is called before the first frame update
    void Awake()
    {
        controlsCanvas.SetActive(false);
    }

    public void ToggleControlsUI()
    {
        controlsCanvas.SetActive(!controlsCanvas.activeInHierarchy);
    }
    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            ToggleControlsUI();
        }
    }
}
