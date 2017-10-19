using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockScript : MonoBehaviour {
    public Vector2 SleepMinMax;

    GameManager myGameManager;
    Renderer ren;
    Collider col;

    private void Awake()
    {
        myGameManager = GetComponentInParent<GameManager>();
        ren = gameObject.GetComponent<Renderer>();
        col = gameObject.GetComponent<Collider>();
    }

    private void OnCollisionEnter(Collision collision)
    {
        myGameManager.BrickHit();
        ren.enabled = false;
        col.enabled = false;
        StartCoroutine(SleepCo(Random.Range(SleepMinMax.x, SleepMinMax.y)));
    }

    IEnumerator SleepCo(float duration)
    {
        yield return new WaitForSeconds(duration);
        Respawn();
    }

    public void Respawn()
    {
        ren.enabled = true;
        col.enabled = true;
    }
}
