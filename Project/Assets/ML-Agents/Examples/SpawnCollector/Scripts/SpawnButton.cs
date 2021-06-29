using UnityEngine;
using Random = UnityEngine.Random;

public class SpawnButton : MonoBehaviour
{
    public Material onMaterial;
    public Material offMaterial;
    public GameObject myButton;
    public GameObject AgentPrefab;
    public SpawnArea Area;
    bool m_State;



    public bool GetState()
    {
        return m_State;
    }


    public void ResetSwitch()
    {
        m_State = false;
        tag = "switchOff";
        transform.rotation = Quaternion.Euler(0f, 0f, 0f);
        myButton.GetComponent<Renderer>().material = offMaterial;
        if (Random.Range(0f, 1f) > 0.9f)
        {
            Activate();
        }
    }

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("agent") && m_State == false)
        {
            Activate();
        }
    }

    void Activate()
    {
        myButton.GetComponent<Renderer>().material = onMaterial;
        m_State = true;
        tag = "switchOn";
        SpawnAgent();
    }

    void SpawnAgent()
    {
        var agent = GameObject.Instantiate(AgentPrefab, gameObject.transform.position + new Vector3(2, 0, 0), default(Quaternion), Area.transform);
        Area.AddReward(0f);
        Area.RegisterAgent(agent);
    }
}
