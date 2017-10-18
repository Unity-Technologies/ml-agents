using UnityEngine;
using System.Collections;

/**
 * This component represents the finish line of a level
*/
public class RunnerFinishLine : MonoBehaviour, IAgentTrigger
{
	public void OnEnter(RunnerAgent agent)
	{
        agent.status = RunnerAgent.AgentStatus.Finished;
	}
}
