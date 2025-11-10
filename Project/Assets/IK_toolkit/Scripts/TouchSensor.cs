using UnityEngine;

public class TouchSensor : MonoBehaviour
{
    public bool isContacted = false;

    void OnTriggerEnter(Collider other)
    {
        if(other.transform.tag == "table")
        {
            isContacted = true;
        }
    }
}
