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

    public bool GetState()
    {
        return state;
    }

    private void Start()
    {
        area = gameObject.transform.parent.gameObject;
        areaComponent = area.GetComponent<PyramidArea>();
        Reset();
    }

    public void Reset()
    {
        transform.position = new Vector3(Random.Range(-areaComponent.range, areaComponent.range), 
                                 2f, Random.Range(-areaComponent.range, areaComponent.range)) 
                             + area.transform.position;
        state = false;
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
            areaComponent.CreatePyramid(1);
            tag = "switchOn";
        }
    }

}
