using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GunController : MonoBehaviour
{
    private bool initialized; //has this robot been initialized
    public KeyCode shootKey = KeyCode.J;

    //SHOOTING RATE
    [Header("SHOOTING RATE")]
    public float shootingRate = .02f; //can shoot every shootingRate seconds. ex: .5 can shoot every .5 seconds
    private float shootTimer;
    public bool coolDownWait;

    //PROJECTILES
    [Header("PROJECTILE")]
    public GameObject projectilePrefab;
    public int numberOfProjectilesToPool = 25;
    public Transform projectileOrigin; //the transform the projectile will originate from
    private List<Projectile> projectilePoolList = new List<Projectile>(); //projectiles to shoot

    //FORCES
    [Header("FORCES")]
    public float forceToUse;
    public ForceMode forceMode;

    // Start is called before the first frame update
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
            p.transform.position = projectileOrigin.position;
            //            p.projectileController = this;
            p.gameObject.SetActive(false);
        }

        initialized = true;
    }

    void Update()
    {
        if (Input.GetKey(shootKey))
        {
            Shoot();
        }
    }

    void FixedUpdate()
    {
        coolDownWait = shootTimer > shootingRate ? false : true;
        shootTimer += Time.fixedDeltaTime;
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
                item.rb.transform.position = projectileOrigin.position;
                item.rb.transform.rotation = projectileOrigin.rotation;

                item.rb.velocity = Vector3.zero;
                item.rb.angularVelocity = Vector3.zero;
                item.rb.gameObject.SetActive(true);
                item.rb.AddForce(projectileOrigin.forward * forceToUse, forceMode);
                break;
            }
        }
    }


}
