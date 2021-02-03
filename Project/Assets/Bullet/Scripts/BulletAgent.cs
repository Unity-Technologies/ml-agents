//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class BulletAgent : MonoBehaviour
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;

    public GameObject area;

    public GameObject bullet;
    Rigidbody m_BulletRb;

    float m_BulletTime;
    /// <summary>
    /// The area bounds.
    /// </summary>
    [HideInInspector]
    public Bounds areaBounds;
    public float rad;

    BulletSettings m_BulletSettings;

    float m_x;
    float m_z;
    float radius;
    float m_currX;
    float m_currZ;
    Vector3 m_center;
    public bool useVectorObs;

    //Rigidbody m_AgentRb;  //cached on initialization
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    EnvironmentParameters m_ResetParams;

    void Awake()
    {
        m_BulletSettings = FindObjectOfType<BulletSettings>();
        areaBounds = ground.GetComponent<Collider>().bounds;
        m_BulletRb = bullet.GetComponent<Rigidbody>();
        //area = GetComponent<BulletArea>();
        m_x = areaBounds.extents.x;
        m_z = areaBounds.extents.z;
        m_currX = area.transform.position.x;
        m_currZ = area.transform.position.z + m_z - rad;
        var angle = (Random.Range(0f,360f) * Mathf.PI/ 180f);
        var x = Mathf.Cos(angle) * (m_currX - area.transform.position.x) - Mathf.Sin(angle) * (m_currZ - area.transform.position.z) + area.transform.position.x;;
        var z = Mathf.Sin(angle) * (m_currX - area.transform.position.x) + Mathf.Cos(angle) * (m_currZ - area.transform.position.z) + area.transform.position.z;
        m_currX = x;
        m_currZ = z;
        m_center = new Vector3(area.transform.position.x, 0.5f, area.transform.position.z);
        m_BulletTime = 0f;
    }


        // Cache the agent rigidbody
     //   m_AgentRb = GetComponent<Rigidbody>();
        // Cache the block rigidbody
        // Get the ground's bounds
        // Get the ground renderer so we can change the material when a goal is scored


    public void FixedUpdate()
    {
        if (Time.time > m_BulletTime + 0.03f)
        {
            //var x = Random.Range(-1f * m_x, m_x) + area.transform.position.x;
            //var z = Random.Range(-1f * m_z, m_z) + area.transform.position.z;
            var r = Random.Range(0f, 360f);
            var ra = Random.Range(0, 2);
            var dir = -1f;
            if (ra == 1)
            {
                dir = 1f;
            }
            var angle = dir * (5f * Mathf.PI/ 180f);
            var x = Mathf.Cos(angle) * (m_currX - area.transform.position.x) - Mathf.Sin(angle) * (m_currZ - area.transform.position.z) + area.transform.position.x;;
            var z = Mathf.Sin(angle) * (m_currX - area.transform.position.x) + Mathf.Cos(angle) * (m_currZ - area.transform.position.z) + area.transform.position.z;
            var pos = new Vector3(x, 0.5f, z);

            Quaternion rotation = Quaternion.Euler(0, r, 0);
            //Quaternion rotation = Quaternion.LookRotation(m_center - pos);
            var ob = Instantiate(m_BulletRb, pos, rotation, area.transform);
            m_currX = x;
            m_currZ = z;
    //        ob.GetComponent<Rigidbody>().AddForce(20f*ob.transform.forward, ForceMode.VelocityChange);
            m_BulletTime = Time.time;

        }
    }
}
