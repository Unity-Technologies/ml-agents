using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomTorqueEveryXSec : MonoBehaviour
{
    public Vector3 maxTorque;
    private Vector3 torqueToUse;
    public float randomizeTorqueEveryXSec = 10;
    public ForceMode forceMode;

    private Rigidbody rb;

    // Start is called before the first frame update
    void OnEnable()
    {
        rb = GetComponent<Rigidbody>();
        InvokeRepeating("SetRandomTorque", 0, randomizeTorqueEveryXSec);
    }

    void SetRandomTorque()
    {
        torqueToUse = new Vector3(Random.Range(-maxTorque.x, maxTorque.x), Random.Range(-maxTorque.y, maxTorque.y),
            Random.Range(-maxTorque.z, maxTorque.z));
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        rb.AddRelativeTorque(torqueToUse, forceMode);
    }
}
