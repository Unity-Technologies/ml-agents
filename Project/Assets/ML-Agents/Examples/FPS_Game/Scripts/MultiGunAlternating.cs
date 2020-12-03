using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

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

    [Header("COOLDOWN & RELOAD")]
    public float CurrentPercentage = 100; //the amount of ammo we currently have on a scale between 0-100
    public float DepletionRate = 5f; //constant rate at which ammo depletes when being used
    public float RegenRate = .25f; //constant rate at which ammo regenerates

    public Slider UISlider;
    // Start is called before the first frame update
    void Start()
    {
        CurrentPercentage = 100;
        if (UISlider)
        {
            UISlider.value = CurrentPercentage;
        }
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
            CurrentPercentage = Mathf.Clamp(CurrentPercentage - DepletionRate, 0, 100);
        }

    }

    void Update()
    {
        if (UISlider)
        {
            UISlider.value = CurrentPercentage;
        }

    }

    void FixedUpdate()
    {
        //        coolDownComplete = shootTimer > shootingRate;
        //        if (coolDownComplete)
        //        {
        if (autoShootEnabled)
        {
            //            ShootGunAtIndex(currentGunIndex);
            Shoot();
        }
        if (AllowKeyboardInput && Input.GetKey(shootKey))
        {
            //            ShootGunAtIndex(currentGunIndex);
            Shoot();
        }
        //        }
        shootTimer += Time.fixedDeltaTime;
        CurrentPercentage = Mathf.Clamp(CurrentPercentage + RegenRate, 0, 100);
    }


    //    IEnumerator HandleShooting()
    //    {
    //        WaitForFixedUpdate wait = new WaitForFixedUpdate();
    //    }
}
