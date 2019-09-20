using UnityEngine;

public class PyramidSwitch : MonoBehaviour
{
    public Material onMaterial;
    public Material offMaterial;
    public GameObject myButton;
    private bool m_State;
    private GameObject m_Area;
    private PyramidArea m_AreaComponent;
    private int m_PyramidIndex;

    public bool GetState()
    {
        return m_State;
    }

    private void Start()
    {
        m_Area = gameObject.transform.parent.gameObject;
        m_AreaComponent = m_Area.GetComponent<PyramidArea>();
    }

    public void ResetSwitch(int spawnAreaIndex, int pyramidSpawnIndex)
    {
        m_AreaComponent.PlaceObject(gameObject, spawnAreaIndex);
        m_State = false;
        m_PyramidIndex = pyramidSpawnIndex;
        tag = "switchOff";
        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        myButton.GetComponent<Renderer>().material = offMaterial;
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("agent") && m_State == false)
        {
            myButton.GetComponent<Renderer>().material = onMaterial;
            m_State = true;
            m_AreaComponent.CreatePyramid(1, m_PyramidIndex);
            tag = "switchOn";
        }
    }
}
