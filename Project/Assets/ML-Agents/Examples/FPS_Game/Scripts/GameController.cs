using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameController : MonoBehaviour
{
    [Header("GLOBAL SETTINGS")]
    public List<Rigidbody> AllRBsList = new List<Rigidbody>();

    public float ExplosionForce = 100;
    public float ExplosionUpwardsModifier = 1;
    public float ExplosionRadius = 10;
    public ForceMode ExplosionForceMode;

    public bool triggerExplosion;

    public enum GameMode
    {
        SinglePlayer, PVP_Single
    }

    public GameMode gameMode;
    [Header("PLAYER DAMAGE")]
    public float DamagePerHit = 15;
    // Start is called before the first frame update
    void Awake()
    {
        Rigidbody[] rbs = Resources.FindObjectsOfTypeAll<Rigidbody>();

        foreach (var rb in rbs)
        {
            if (!rb.transform.CompareTag("projectile"))
            {
                AllRBsList.Add(rb);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (triggerExplosion)
        {
            triggerExplosion = false;
            AddExplosiveForcesToAllRB(transform.position);
        }

    }

    public void AddExplosiveForcesToAllRB(Vector3 pos)
    {
        foreach (var rb in AllRBsList)
        {
            rb.AddExplosionForce(ExplosionForce, pos, ExplosionRadius, ExplosionUpwardsModifier, ExplosionForceMode);
        }
    }
}
