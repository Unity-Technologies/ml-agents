using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerLookAt : MonoBehaviour
{
    public float speed = 5f;

    void Update()
    {

        float horizontal = Input.GetAxis("Mouse X") * speed;
        float vertical = Input.GetAxis("Mouse Y") * speed;

        transform.Rotate(0f, horizontal, 0f, Space.World);
        transform.Rotate(-vertical, 0f, 0f, Space.Self);

    }
}
