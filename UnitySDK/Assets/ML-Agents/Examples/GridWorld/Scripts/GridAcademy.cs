using UnityEngine;
using MLAgents;

public class GridAcademy : Academy
{
    public Camera MainCamera;

    public override void InitializeAcademy()
    {
        FloatProperties.RegisterCallback("gridSize", f =>
        {
            MainCamera.transform.position = new Vector3(-(f - 1) / 2f, f * 1.25f, -(f - 1) / 2f);
            MainCamera.orthographicSize = (f + 5f) / 2f;
        });

    }
}
