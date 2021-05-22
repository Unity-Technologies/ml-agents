//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class Bullet : MonoBehaviour
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public Rigidbody BulletRb;
    public float speed;
    void Awake()
    {
        BulletRb = GetComponent<Rigidbody>();
        //BulletRb.AddForce(20f*transform.forward, ForceMode.VelocityChange);
    }

    public void FixedUpdate()
    {
        transform.position += transform.forward * speed;
    }
    public void OnCollisionEnter(Collision c)
    {
        if (c.gameObject.CompareTag("wall"))
        {
            //gameObject.SetActive(false);
            Destroy(gameObject);
        }
    }
}
