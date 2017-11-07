using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeathZoneScript : MonoBehaviour {

    GameManager myGameManager;

    private void Start()
    {
        myGameManager = GetComponentInParent<GameManager>();
    }

    private void OnTriggerEnter(Collider other)
    {
        other.gameObject.SetActive(false);
        myGameManager.BallLost();
    }
}
