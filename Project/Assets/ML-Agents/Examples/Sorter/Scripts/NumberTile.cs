using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NumberTile : MonoBehaviour
{
    public int NumberValue;
    public Material DefaultMaterial;
    public Material SuccessMaterial;

    private bool m_Visited = false;
    private MeshRenderer m_Renderer;

    void Awake()
    {
        m_Renderer = GetComponentInChildren<MeshRenderer>();
        ResetTile();
    }

    public bool IsVisited
    {
        get { return m_Visited; }
    }

    public void VisitTile()
    {
        m_Renderer.sharedMaterial = SuccessMaterial;
        m_Visited = true;
    }

    public void ResetTile()
    {
        m_Renderer.sharedMaterial = DefaultMaterial;
        m_Visited = false;
    }
}
