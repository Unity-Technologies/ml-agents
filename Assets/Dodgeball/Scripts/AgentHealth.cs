using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using Cinemachine;

public class AgentHealth : MonoBehaviour
{
    public float CurrentPercentage = 100;
    public Slider UISlider;

    public MeshRenderer bodyMesh;
    public Color damageColor;
    public Color startingColor;
    public float damageFlashDuration = .02f;

    public ShieldController ShieldController;

    public GameObject CubeBody;
    public GameObject DeathCube;
    public GameObject ExplosionParticles;
    public CinemachineImpulseSource impulseSource;
    public bool ResetSceneAfterDeath = false;
    public bool Dead;

    // private GameController GameController;
    [Header("PLAYER DAMAGE")] public bool UseGlobalDamageSettings;
    public float DamagePerHit = 15f; //constant rate at which ammo depletes when being used

    private Rigidbody rb;
    // Start is called before the first frame update
    void OnEnable()
    {
        // GameController = FindObjectOfType<GameController>();
        CurrentPercentage = 100;
        if (UISlider)
        {
            UISlider.value = CurrentPercentage;
        }

        if (bodyMesh)
        {
            startingColor = bodyMesh.sharedMaterial.color;
        }

        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Dead)
        {
            return;
        }
        if (UISlider)
        {
            UISlider.value = CurrentPercentage;
        }
    }

    private void OnCollisionEnter(Collision col)
    {
        if (Dead)
        {
            return;
        }
        if (col.transform.CompareTag("projectile"))
        {
            if (ShieldController && ShieldController.ShieldIsActive)
            {
                return;
            }

            var damage = DamagePerHit;
            if (UseGlobalDamageSettings)
            {
                // damage = GameController.DamagePerHit;
            }
            CurrentPercentage = Mathf.Clamp(CurrentPercentage - damage, 0, 100);
            if (CurrentPercentage == 0)
            {
                Dead = true;
                rb.isKinematic = true;
                //                rb.velocity = Vector3.zero;
                //                rb.angularVelocity = Vector3.zero;
                CubeBody.SetActive(false);
                DeathCube.transform.position = CubeBody.transform.position;
                DeathCube.SetActive(true);
                ExplosionParticles.transform.position = CubeBody.transform.position;
                ExplosionParticles.SetActive(true);
                // if (GameController)
                // {
                //     GameController.AddExplosiveForcesToAllRB(CubeBody.transform.position);
                // }
                if (impulseSource)
                {
                    impulseSource.GenerateImpulse();
                }
                if (ResetSceneAfterDeath)
                {
                    StartCoroutine(RestartScene());
                }
            }
            StartCoroutine(BodyDamageFlash());
        }
    }

    //    IEnumerator Explosion()
    //    {
    //        if (impulseSource)
    //        {
    //            impulseSource.GenerateImpulse();
    //        }
    //        WaitForFixedUpdate wait = new WaitForFixedUpdate();
    //        float timer = 0;
    //        while (timer < 3)
    //        {
    //            timer += Time.fixedDeltaTime;
    //            yield return wait;
    //        }
    //    }

    IEnumerator RestartScene()
    {

        //        if (impulseSource)
        //        {
        //            impulseSource.GenerateImpulse();
        //        }
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        float timer = 0;
        while (timer < 1)
        {
            timer += Time.fixedDeltaTime;
            yield return wait;
        }
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
    IEnumerator BodyDamageFlash()
    {
        WaitForFixedUpdate wait = new WaitForFixedUpdate();
        if (bodyMesh)
        {
            bodyMesh.material.color = damageColor;
        }
        float timer = 0;
        while (timer < damageFlashDuration)
        {
            timer += Time.fixedDeltaTime;
            yield return wait;
        }
        if (bodyMesh)
        {
            bodyMesh.material.color = startingColor;
        }
    }
}
