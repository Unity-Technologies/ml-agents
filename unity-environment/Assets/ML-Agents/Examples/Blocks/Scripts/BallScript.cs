using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallScript : MonoBehaviour {

    public float InitialForce = 300;
    public float MinimumMagnitude = 5f;
    public Vector3 RandomForceMax = Vector3.zero;
    public Vector2 StartRandomization;
    Rigidbody body;
    Vector3 startPosition;

    private void Awake()
    {
        startPosition = transform.position;
        body = gameObject.GetComponent<Rigidbody>();
    }

    // Use this for initialization
    void Start () {
        RandomStartPosition();
        AddRandomForce();
	}

    public void RandomStartPosition()
    {
        transform.position = startPosition + new Vector3(Random.Range(-StartRandomization.x, StartRandomization.x), Random.Range(-StartRandomization.y, StartRandomization.y), 0);
    }

    void AddRandomForce()
    {
        body.AddForce(new Vector3(Random.Range(-RandomForceMax.x, RandomForceMax.x), Random.Range(-RandomForceMax.y, RandomForceMax.y), Random.Range(-RandomForceMax.z, RandomForceMax.z)));
    }

    private void OnCollisionEnter(Collision collision)
    {
        AddRandomForce();
    }

    private void Update()
    {
        if (body.velocity.magnitude < MinimumMagnitude)
        {
            body.AddForce(transform.up * InitialForce);
        }
    }

}
