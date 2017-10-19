using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentBuilderScript : MonoBehaviour {

    public GameObject prefab;
    public int rows;
    public int columns;
    public float rowSpacing;
    public float columnSpacing;
    public Brain BrainToUse;

    private void Awake()
    {
        Vector2 cursor = new Vector2(columnSpacing * columns * -0.5f, rowSpacing * rows * -0.5f);

        for (var y = 0; y < rows; y++)
        {
            for (var x = 0; x < columns; x++)
            {
                var env = Instantiate<GameObject>(prefab, cursor, Quaternion.identity);
                var pas = env.GetComponentInChildren<PaddleAgentScript>();
                pas.GiveBrain(BrainToUse);
                cursor.x += columnSpacing;
                
            }
            cursor.x = columnSpacing * columns * -0.5f;
            cursor.y += rowSpacing;
        }
    }

    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
