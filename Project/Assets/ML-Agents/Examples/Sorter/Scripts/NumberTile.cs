using UnityEngine;

public class NumberTile : MonoBehaviour
{
    public int NumberValue;
    public Material DefaultMaterial;
    public Material SuccessMaterial;

    private bool m_Visited;
    private MeshRenderer m_Renderer;

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
        if (m_Renderer is null)
        {
            m_Renderer = GetComponentInChildren<MeshRenderer>();
        }
        m_Renderer.sharedMaterial = DefaultMaterial;
        m_Visited = false;
    }
}
