using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Logic for robot projectiles.
/// </summary>
public class ShootProjectiles : MonoBehaviour
{
    public bool initialized; //has this robot been initialized
    public GameObject projectilePrefab;
    public int numberOfProjectilesToPool = 25;

    private List<Projectile> projectilePoolList = new List<Projectile>(); //projectiles to shoot
    public Transform projectileStartingPos; //the transform the projectile will originate from
    public float projectileLaunchAngle = 5; //the angle at which the projectile will launch
    public float shootingRate = .02f; //can shoot every shootingRate seconds. ex: .5 can shoot every .5 seconds
    private float shootTimer;
    public bool coolDownWait;

    //for standalone projectiles
    public bool autoShootEnabled;
    public float autoShootDistance = 30;

    public bool useStandaloneInput = false;
    public KeyCode shootKey = KeyCode.J;
    void Start()
    {
        if (!initialized)
        {
            Initialize();
        }
    }

    void OnEnable()
    {
        if (!initialized)
        {
            Initialize();
        }
    }


    void Initialize()
    {
        projectilePoolList.Clear(); //clear list in case it's not empty
        for (var i = 0; i < numberOfProjectilesToPool; i++)
        {
            GameObject obj = Instantiate(projectilePrefab, transform.position, Quaternion.identity);
            Projectile p = obj.GetComponent<Projectile>();
            projectilePoolList.Add(p);
            p.transform.position = projectileStartingPos.position;
            //            p.projectileController = this;
            p.gameObject.SetActive(false);
        }

        initialized = true;
    }

    void Update()
    {
        if (Input.GetKey(shootKey))
        {
            Shoot(projectileStartingPos.position,
                projectileStartingPos.TransformPoint(Vector3.forward * autoShootDistance));
        }
    }

    void FixedUpdate()
    {
        coolDownWait = shootTimer > shootingRate ? false : true;
        shootTimer += Time.fixedDeltaTime;
        if (autoShootEnabled)
        {
            Shoot(projectileStartingPos.position,
                projectileStartingPos.TransformPoint(Vector3.forward * autoShootDistance));
            Debug.DrawRay(projectileStartingPos.TransformPoint(Vector3.forward * autoShootDistance), Vector3.up);
        }
    }


    public void Shoot(Vector3 startPos, Vector3 targetPos)
    {
        if (coolDownWait)
        {
            return;
        }

        shootTimer = 0; //reset timer

        //shoot first available projectile in the pool
        foreach (var item in projectilePoolList)
        {
            if (!item.gameObject.activeInHierarchy)
            {
                LaunchProjectile(item.rb, startPos, targetPos); //shoot
                break;
            }
        }
    }
    public void Shoot()
    {
        if (coolDownWait)
        {
            return;
        }

        shootTimer = 0; //reset timer

        //shoot first available projectile in the pool
        foreach (var item in projectilePoolList)
        {
            if (!item.gameObject.activeInHierarchy)
            {
                LaunchProjectile(item.rb, projectileStartingPos.position, projectileStartingPos.TransformPoint(Vector3.forward * autoShootDistance));
                break;
            }
        }
    }


    public void LaunchProjectile(Rigidbody rb, Vector3 startPos, Vector3 targetPos)
    {
        rb.transform.position = startPos;
        //        rb.transform.rotation = Quaternion.identity;
        rb.transform.rotation = transform.rotation;

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.gameObject.SetActive(true);
        Vector3 p = targetPos;

        float gravity = Physics.gravity.magnitude;

        // Selected angle in radians
        float angle = projectileLaunchAngle * Mathf.Deg2Rad;

        // Positions of this object and the target on the same plane
        Vector3 planarTarget = new Vector3(p.x, 0, p.z);
        Vector3 planarPostion = new Vector3(startPos.x, 0, startPos.z);

        // Planar distance between objects
        float distance = Vector3.Distance(planarTarget, planarPostion);
        // Distance along the y axis between objects
        float yOffset = startPos.y - p.y;

        float initialVelocity = (1 / Mathf.Cos(angle)) *
                                Mathf.Sqrt((0.5f * gravity * Mathf.Pow(distance, 2)) /
                                           (distance * Mathf.Tan(angle) + yOffset));

        Vector3 velocity = new Vector3(0, initialVelocity * Mathf.Sin(angle), initialVelocity * Mathf.Cos(angle));

        // Rotate our velocity to match the direction between the two objects
        float angleBetweenObjects =
            Vector3.Angle(Vector3.forward, planarTarget - planarPostion) * (p.x > startPos.x ? 1 : -1);
        Vector3 finalVelocity = Quaternion.AngleAxis(angleBetweenObjects, Vector3.up) * velocity;
        if (!float.IsNaN(finalVelocity.x)) //NaN checked
        {
            rb.velocity = finalVelocity;
        }
    }
}
