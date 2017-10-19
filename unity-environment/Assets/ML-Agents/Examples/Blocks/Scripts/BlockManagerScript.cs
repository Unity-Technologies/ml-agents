using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockManagerScript : MonoBehaviour {

    public int columns;
    public int rows;
    public Vector2 spacing = new Vector2(1.2f, -0.45f);
    public Vector3 startingPosititon = new Vector3(-1.8f, 4.2f, 0f);
    public GameObject prefab;

	// Use this for initialization
	void Start () {
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                Instantiate<GameObject>(prefab, transform.position + startingPosititon + new Vector3(spacing.x * column, spacing.y * row, 0), Quaternion.identity, transform);
            }
        }
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
