using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BananaLogic : MonoBehaviour {

    public bool respawn;
    public BananaArea myArea;

    // Use this for initialization
    void Start () {
        
    }
    
    // Update is called once per frame
    void Update () {
        
    }

    public void OnEaten() {
        if (respawn) 
        {
            transform.position = new Vector3(Random.Range(-myArea.range, myArea.range), 
                                             transform.position.y + 3f, 
                                             Random.Range(-myArea.range, myArea.range));
        }
        else 
        {
            Destroy(gameObject);
        }
    }
}
