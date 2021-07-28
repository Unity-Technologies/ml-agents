using UnityEngine;
using Random = UnityEngine.Random;
using Unity.MLAgents;

public class SpawnButton : MonoBehaviour
{
    public Material onMaterial;
    public Material offMaterial;
    public GameObject myButton;
    public GameObject AgentPrefab;
    public SpawnArea Area;
    bool m_State;

    int m_ResetTimer;



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
            Activate(null);
        }
    }

    void FixedUpdate(){
        if (m_State)
        {
            m_ResetTimer -= 1;
            if (m_ResetTimer < 0)
            {
                ResetSwitch();
            }
        }
    }

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("agent") && m_State == false)
        {
            Activate(other.gameObject);
        }
    }

    void Activate(GameObject pressingAgent)
    {
        myButton.GetComponent<Renderer>().material = onMaterial;
        m_State = true;
        tag = "switchOn";
        SpawnAgent(pressingAgent);
        m_ResetTimer = 250;
    }

    void SpawnAgent(GameObject pressingAgent)
    {
        // if (pressingAgent != null)
        // {
        //     Area.UnregisterAgent(pressingAgent);
        //     Destroy(pressingAgent);
        //     var agent1 = GameObject.Instantiate(AgentPrefab, gameObject.transform.position + new Vector3(2 , 0, 0), default(Quaternion), Area.transform);
        //     // Area.AddReward(0f);
        //     Area.RegisterAgent(agent1);
        // }

        var agent2 = GameObject.Instantiate(AgentPrefab, gameObject.transform.position + new Vector3(3, 0, 0), default(Quaternion), Area.transform);
        // Area.AddReward(0f);
        Area.RegisterAgent(agent2);
    }
}
