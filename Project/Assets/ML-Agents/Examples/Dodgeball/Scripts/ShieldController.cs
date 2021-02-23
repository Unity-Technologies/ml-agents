using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShieldController : MonoBehaviour
{
    public KeyCode shieldKey = KeyCode.I;

    public GameObject shieldGO;

    public bool ShieldIsActive;
    [Header("COOLDOWN & REGEN")]
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

    // Update is called once per frame
    void Update()
    {
        //        if (Input.GetKey(shieldKey))
        //        {
        //            ActivateShield(true);
        ////            if (!shieldGO.activeInHierarchy)
        ////            {
        ////                shieldGO.SetActive(true);
        ////                ShieldIsActive = true;
        ////            }
        //        }
        //        else
        //        {
        //            ActivateShield(false);
        ////            if (shieldGO.activeInHierarchy)
        ////            {
        ////                shieldGO.SetActive(false);
        ////                ShieldIsActive = false;
        ////            }
        //        }
        if (UISlider)
        {
            UISlider.value = CurrentPercentage;
        }
    }

    public void ActivateShield(bool shouldBeActive)
    {

        if (!shieldGO.activeInHierarchy && shouldBeActive)
        {
            shieldGO.SetActive(true);
            ShieldIsActive = true;
        }
        if (shieldGO.activeInHierarchy && !shouldBeActive)
        {
            shieldGO.SetActive(false);
            ShieldIsActive = false;
        }

        if (ShieldIsActive)
        {
            CurrentPercentage = Mathf.Clamp(CurrentPercentage - DepletionRate, 0, 100);
        }
        else
        {
            CurrentPercentage = Mathf.Clamp(CurrentPercentage + RegenRate, 0, 100);
        }
    }
}
