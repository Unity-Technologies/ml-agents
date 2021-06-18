using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlockTrigger : MonoBehaviour
{
    public WallJumpAgent agent;
    // Start is called before the first frame update
    void OnCollisionStay(Collision col)
    {
        if (col.gameObject.CompareTag("wall") && agent.taskType == WallJumpAgent.TaskType.PushBlock)
        {
            agent.AgentWon();
        }
    }
}
