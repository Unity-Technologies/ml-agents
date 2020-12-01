using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MultiGunAlternating : MonoBehaviour
{

    [Header("INPUT")] public bool AllowKeyboardInput = true;
    public KeyCode shootKey = KeyCode.J;

    [Header("AUTOSHOOT")] public bool autoShootEnabled;

    [Header("GUNS")] public List<GunController> gunList = new List<GunController>();
    public int currentGunIndex;

    [Header("TIMING")]
    public float shootingRate = .02f; //can shoot every shootingRate seconds. ex: .5 can shoot every .5 seconds
    private float shootTimer;
    public bool coolDownComplete;

    public
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

        //        if(shoot)
        //        while (Input.GetKey(shootKey))
        //        {
        ////            Shoot();
        //            ShootQuantity(1);
        //        }
        //        if (Input.GetKeyDown(shootKey))
        //        {
        ////            Shoot();
        //            ShootQuantity(1);
        //        }
    }

    void ShootGunAtIndex(int i)
    {
        currentGunIndex = i >= gunList.Count - 1 ? 0 : i + 1;
        gunList[currentGunIndex].Shoot();
        shootTimer = 0;
    }

    public void Shoot()
    {
        coolDownComplete = shootTimer > shootingRate;
        if (coolDownComplete)
        {
            ShootGunAtIndex(currentGunIndex);
        }
    }

    void FixedUpdate()
    {
        //        coolDownComplete = shootTimer > shootingRate;
        //        if (coolDownComplete)
        //        {
        if (autoShootEnabled)
        {
            ShootGunAtIndex(currentGunIndex);
        }
        if (AllowKeyboardInput && Input.GetKey(shootKey))
        {
            ShootGunAtIndex(currentGunIndex);
        }
        //        }
        shootTimer += Time.fixedDeltaTime;
    }


    //    IEnumerator HandleShooting()
    //    {
    //        WaitForFixedUpdate wait = new WaitForFixedUpdate();
    //    }
}
