using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleScript : MonoBehaviour {

    public float speed;
    public KeyCode leftKey;
    public KeyCode rightKey;

    Vector3 startPosition;
    float limit = 3f;

    private void Awake()
    {
        startPosition = transform.position;
    }

    // Update is called once per frame
    void Update () {
        if (Input.GetKey(leftKey)) Left();
        if (Input.GetKey(rightKey)) Right();
    }

    public void Left()
    {
        transform.Translate(-speed * Time.deltaTime, 0, 0);
        transform.position = new Vector3(Mathf.Clamp(transform.position.x, startPosition.x - limit, startPosition.x + limit), startPosition.y, startPosition.z);
    }

    public void Right()
    {
        transform.Translate(speed * Time.deltaTime, 0, 0);
        transform.position = new Vector3(Mathf.Clamp(transform.position.x, startPosition.x - limit, startPosition.x + limit), startPosition.y, startPosition.z);
    }
}
