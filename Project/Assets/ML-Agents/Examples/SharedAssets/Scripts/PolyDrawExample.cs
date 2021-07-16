using System;
using UnityEngine;
using System.Collections.Generic;
using UnityEditor;

public class PolyDrawExample : MonoBehaviour
{
    public int numberOfSides;
    public float polygonRadius;
    // public Vector2 polygonCenter;

    void Update()
    {
        // DebugDrawPolygon(polygonCenter, polygonRadius, numberOfSides);
        DebugDrawPolygon(transform.position, polygonRadius, numberOfSides);
    }

    private void OnDrawGizmosSelected()
    {
        foreach (var pos in rayPosList)
        {
            Gizmos.DrawSphere(pos,.1f);
        }
    }

    public List<Vector3> rayPosList = new List<Vector3>();
    // Draw a polygon in the XY plane with a specfied position, number of sides
    // and radius.
    void DebugDrawPolygon(Vector3 center, float radius, int numSides)
    {
        rayPosList.Clear();
        // The corner that is used to start the polygon (parallel to the X axis).
        // Vector3 startCorner = new Vector3(radius, 0) + center;
        // Vector3 startCorner = new Vector3(radius, 0, radius);
        Vector3 startCorner = transform.TransformPoint(new Vector3(radius, 0, radius));
        startCorner.y = transform.position.y;

        // The "previous" corner point, initialised to the starting corner.
        Vector3 previousCorner = startCorner;

        // For each corner after the starting corner...
        for (int i = 0; i < numSides; i++)
        {
            // Calculate the angle of the corner in radians.
            float cornerAngle = 2f * Mathf.PI / (float)numSides * i;

            // Get the X and Y coordinates of the corner point.
            // Vector2 currentCorner = new Vector2(Mathf.Cos(cornerAngle) * radius, Mathf.Sin(cornerAngle) * radius) + center;
            // Vector2 currentCorner = new Vector2(Mathf.Sin(cornerAngle) * radius, Mathf.Cos(cornerAngle) * radius) + center;
            Vector3 currentCorner = new Vector3(Mathf.Cos(cornerAngle) * radius, 0, Mathf.Sin(cornerAngle) * radius);
            currentCorner = transform.TransformPoint(currentCorner);
            currentCorner.y = transform.position.y;
            rayPosList.Add(currentCorner);
            // Debug.DrawRay(currentCorner);
            // Draw a side of the polygon by connecting the current corner to the previous one.
            Debug.DrawLine(currentCorner, previousCorner);

            // Having used the current corner, it now becomes the previous corner.
            previousCorner = currentCorner;
        }

        // Draw the final side by connecting the last corner to the starting corner.
        Debug.DrawLine(startCorner, previousCorner);
    }
}
