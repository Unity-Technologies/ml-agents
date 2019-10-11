using UnityEngine;
using MLAgents;

public class GridAcademy : Academy
{
    public Camera MainCamera;

    public override void AcademyReset()
    {
        MainCamera.transform.position = new Vector3(-((int)resetParameters["gridSize"] - 1) / 2f,
            (int)resetParameters["gridSize"] * 1.25f,
            -((int)resetParameters["gridSize"] - 1) / 2f);
        MainCamera.orthographicSize = ((int)resetParameters["gridSize"] + 5f) / 2f;
    }
}
