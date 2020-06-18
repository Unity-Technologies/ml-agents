using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UILookAtCamera : MonoBehaviour
{
    private Camera m_camera;

    public bool inverseLookDir;
    // Start is called before the first frame update
    void Start()
    {
        m_camera = Camera.main;
    }

    // Update is called once per frame
    void Update()
    {

        var lookDir = m_camera.transform.position - transform.position;
        lookDir *= inverseLookDir ? -1 : 1;
        transform.rotation = Quaternion.LookRotation(lookDir);
//        transform.LookAt(m_camera.transform, m_camera.transform.up);
    }
}
