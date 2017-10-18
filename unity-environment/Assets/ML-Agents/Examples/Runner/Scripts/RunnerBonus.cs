using System.Collections;
using UnityEngine;

/**
 * This component represents a collectable bonus
*/
public class RunnerBonus : MonoBehaviour, IAgentTrigger
{
    public void FixedUpdate()
    {
        transform.localRotation = transform.localRotation * Quaternion.AngleAxis(-2, Vector3.up);
    }

    public void OnEnter(RunnerAgent agent)
    {
        agent.collectedBonus++;

        GetComponent<Collider>().enabled = false;
        StartCoroutine(DestroyedAnimation());
    }

    private IEnumerator DestroyedAnimation()
    {
        for (int frame = 0; frame < 10; frame++)
        {
            transform.position = transform.position + Vector3.up * 0.1f;
            transform.localScale = transform.localScale * 1.01f;
            yield return new WaitForEndOfFrame();
        }

        Destroy(gameObject);
    }
}
