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
        var count = 1;
        Vector2 placementCursor = new Vector2(columnSpacing * columns * -0.5f, rowSpacing * rows * -0.5f);

        for (var y = 0; y < rows; y++)
        {
            for (var x = 0; x < columns; x++)
            {
                var env = Instantiate<GameObject>(prefab, placementCursor, Quaternion.identity, transform);
                env.name = "Environment " + count++;

                var pas = env.GetComponentInChildren<Agent>();
                pas.GiveBrain(BrainToUse);
                placementCursor.x += columnSpacing;
                
            }
            placementCursor.x = columnSpacing * columns * -0.5f;
            placementCursor.y += rowSpacing;
        }
    }


}
