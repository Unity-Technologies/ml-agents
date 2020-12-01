using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cinemachine;

public class GunController : MonoBehaviour
{
    public bool AllowKeyboardInput = true; //this mode ignores player input
    private bool initialized; //has this robot been initialized
    public KeyCode shootKey = KeyCode.J;
    [Header("AUTOSHOOT")] public bool autoShootEnabled;


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

    [Header("SCREEN SHAKE")]
    public bool UseScreenShake;

    [Header("TRANSFORM SHAKE")] public bool ShakeTransform;
    public float ShakeDuration = .1f;
    public float ShakeAmount = .1f;
    private Vector3 startPos;
    private bool m_TransformIsShaking;

    CinemachineImpulseSource impulseSource;

    // Start is called before the first frame update
    void Awake()
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
            impulseSource = GetComponent<CinemachineImpulseSource>();
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
        if (!AllowKeyboardInput)
        {
            return;
        }
        if (Input.GetKeyDown(shootKey))
        {
            Shoot();
            //            ShootQuantity(1);
        }
    }

    void FixedUpdate()
    {
        coolDownWait = shootTimer > shootingRate ? false : true;
        shootTimer += Time.fixedDeltaTime;
        if (autoShootEnabled)
        {
            Shoot();
        }
    }

    //    public void ShootQuantity(int num)
    //    {
    //        var i = 0;
    //        while (i < num)
    //        {
    //            //shoot first available projectile in the pool
    //            foreach (var item in projectilePoolList)
    //            {
    //                if (!item.gameObject.activeInHierarchy)
    //                {
    //                    FireProjectile(item.rb);
    //                    impulseSource.GenerateImpulse();
    //                    break;
    //                }
    //            }
    //
    //            i++;
    //        }
    //    }

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
                FireProjectile(item.rb);
                if (UseScreenShake && impulseSource)
                {
                    impulseSource.GenerateImpulse();
                }
                if (ShakeTransform && !m_TransformIsShaking)
                {
                    StartCoroutine(I_ShakeTransform());
                }
                break;
            }
        }

    }

    public void FireProjectile(Rigidbody rb)
    {
        rb.transform.position = projectileOrigin.position;
        rb.transform.rotation = projectileOrigin.rotation;

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.gameObject.SetActive(true);
        rb.AddForce(projectileOrigin.forward * forceToUse, forceMode);
    }

    IEnumerator I_ShakeTransform()
    {
        m_TransformIsShaking = true;
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        float timer = 0;
        startPos = transform.localPosition;
        while (timer < ShakeDuration)
        {
            var pos = startPos + (Random.insideUnitSphere * ShakeAmount);
            transform.localPosition = pos;
            timer += Time.fixedDeltaTime;
            yield return wait;
        }
        transform.localPosition = startPos;
        m_TransformIsShaking = false;
    }

}
