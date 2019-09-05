using UnityEngine;

public class ReacherGoal : MonoBehaviour
{
    public GameObject agent;
    public GameObject hand;
    public GameObject goalOn;

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == hand)
        {
            goalOn.transform.localScale = new Vector3(1f, 1f, 1f);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject == hand)
        {
            goalOn.transform.localScale = new Vector3(0f, 0f, 0f);
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.gameObject == hand)
        {
            agent.GetComponent<ReacherAgent>().AddReward(0.01f);
        }
    }
}
