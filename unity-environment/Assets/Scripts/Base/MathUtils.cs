using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 2D Vector of integer
/// </summary>
[System.Serializable]
public struct Vector2i
{
    public Vector2i(int x = 0, int y = 0)
    {
        this.x = x; this.y = y;
    }

    public int ManhattanDistanceTo(Vector2i goal)
    {
        return Mathf.Abs(goal.x - x) + Mathf.Abs(goal.y - y);
    }

    public bool Equals(Vector2i v)
    {
        return v.x == x && v.y == y;
    }
    public int x, y;
}