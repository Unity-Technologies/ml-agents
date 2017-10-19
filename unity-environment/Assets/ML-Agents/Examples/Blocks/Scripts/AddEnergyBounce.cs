using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AddEnergyBounce : MonoBehaviour {

    public Vector3 additionalForce;

    private void OnCollisionEnter(Collision collision)
    {
        collision.rigidbody.AddForce(additionalForce);
    }
}