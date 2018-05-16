using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PyramidSwitch : MonoBehaviour
{
    public Material onMaterial;
    public Material offMaterial;
    public GameObject myButton;
    private bool state;
    private GameObject area;
    private PyramidArea areaComponent;
    private int pyramidIndex;

    public bool GetState()
    {
        return state;
    }

    private void Start()
    {
        area = gameObject.transform.parent.gameObject;
        areaComponent = area.GetComponent<PyramidArea>();
    }

    public void ResetSwitch(int spawnAreaIndex, int pyramidSpawnIndex)
    {
        areaComponent.PlaceObject(gameObject, spawnAreaIndex);
        state = false;
        pyramidIndex = pyramidSpawnIndex;
        tag = "switchOff";
        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        myButton.GetComponent<Renderer>().material = offMaterial;
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("agent") && state == false)
        {
            myButton.GetComponent<Renderer>().material = onMaterial;
            state = true;
            areaComponent.CreatePyramid(1, pyramidIndex);
            tag = "switchOn";
        }
    }

}
