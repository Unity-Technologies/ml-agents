using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/** Generic functions for parent Agent class.
 * Contains all logic for Brain-Agent communication and Agent-Environment 
 * interaction.
 */
public abstract class Agent : MonoBehaviour
{
    public Brain brain;
    /**<  \brief  The brain that will control this agent. */
    /**< Use the inspector to drag the desired brain gameObject into
	 * the Brain field */

    public List<Camera> observations;
    /**<  \brief  The list of the cameras the Agent uses as observations. */
    /**< These cameras will be used to generate the observations */

    public int maxStep;
    /**<  \brief  The number of steps the agent takes before being done. */
    /**< If set to 0, the agent can only be set to done via a script.
    * If set to any positive integer, the agent will be set to done after that
    * many steps each episode. */

    public bool resetOnDone = true;
    /**<  \brief Determines the behaviour of the Agent when done.*/
    /**< If true, the agent will reset when done. 
	 * If not, the agent will remain done, and no longer take actions.*/

    [HideInInspector]
    public float reward;
    /**< \brief Describes the reward for the given step of the agent.*/
    /**< It is reset to 0 at the beginning of every step. 
	* Modify in AgentStep(). 
	* Should be set to positive to reinforcement desired behavior, and
	* set to a negative value to punish undesireable behavior.
    * Additionally, the magnitude of the reward should not exceed 1.0 */

    [HideInInspector]
    public bool done;
    /**< \brief Whether or not the agent is done*/
    /**< Set to true when the agent has acted in some way which ends the 
	 * episode for the given agent. */

    [HideInInspector]
    public float value;
    /**< \brief The current value estimate of the agent */
    /**<  When using an External brain, you can pass value estimates to the
	 * agent at every step using env.Step(actions, values).
	 * If AgentMonitor is attached to the Agent, this value will be displayed.*/

    [HideInInspector]
    public float CumulativeReward;
    /**< \brief Do not modify: This keeps track of the cumulative reward.*/

    [HideInInspector]
    public int stepCounter;
    /**< \brief Do not modify: This keeps track of the number of steps taken by
     * the agent each episode.*/

    [HideInInspector]
    public float[] agentStoredAction;
    /**< \brief Do not modify: This keeps track of the last actions decided by
     * the brain.*/

    [HideInInspector]
    public float[] memory;
    /**< \brief Do not modify directly: This is used by the brain to store 
     * information about the previous states of the agent*/

    [HideInInspector]
    public int id;
    /**< \brief Do not modify : This is the unique Identifier each agent 
     * receives at initialization. It is used by the brain to identify
     * the agent.*/

    void OnEnable()
    {
        id = gameObject.GetInstanceID();
        if (brain != null)
        {
            brain.agents.Add(id, gameObject.GetComponent<Agent>());
            if (brain.brainParameters.actionSpaceType == StateType.continuous)
            {
                agentStoredAction = new float[brain.brainParameters.actionSize];
            }
            else
            {
                agentStoredAction = new float[1];
            }
            memory = new float[brain.brainParameters.memorySize];
        }
        InitializeAgent();
    }

    void OnDisable()
    {
        //Remove the agent from the list of agents of the brain
        brain.agents.Remove(id);
    }

    /// When GiveBrain is called, the agent unsubscribes from its 
    /// previous brain and subscribes to the one passed in argument.
    /** Use this method to provide a brain to the agent via script. 
	 * Do not modify brain directly.
	@param b The Brain component the agent will subscribe to.*/
    public void GiveBrain(Brain b)
    {
        RemoveBrain();
        brain = b;
        brain.agents.Add(id, gameObject.GetComponent<Agent>());
        if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            agentStoredAction = new float[brain.brainParameters.actionSize];
        }
        else
        {
            agentStoredAction = new float[1];
        }
        memory = new float[brain.brainParameters.memorySize];
    }

    /// When RemoveBrain is called, the agent unsubscribes from its brain.
    /** Use this method to give a brain to an agent via script. 
	 * Do not modify brain directly.
	 * If an agent does not have a brain, it will not update its actions.*/
    public void RemoveBrain()
    {
        if (brain != null)
        {
            brain.agents.Remove(id);
        }
    }

    /// Initialize the agent with this method
    /** Must be implemented in agent-specific child class.
	 *  This method called only once when the agent is enabled.
	*/
    public virtual void InitializeAgent()
    {

    }

    /// Collect the states of the agent with this method
    /** Must be implemented in agent-specific child class.
	 *  This method called at every step and collects the state of the agent.
	 *  The lenght of the output must be the same length as the state size field
	 *  in the brain parameters of the brain the agent subscribes to.
	 *  Note : The order of the elements in the state list is important.
	 *  @returns state A list of floats corresponding to the state of the agent. 
	*/
    public virtual List<float> CollectState()
    {
        List<float> state = new List<float>();
        return state;
    }

    /// Defines agent-specific behavior at every step depending on the action.
    /** Must be implemented in agent-specific child class.
	 *  Note: If your state is discrete, you need to convert your 
	 *  state into a list of float with length 1.
	 *  @param action The action the agent receives from the brain. 
	*/
    public virtual void AgentStep(float[] action)
    {

    }


    /// Defines agent-specific behaviour when done
    /** Must be implemented in agent-specific child class. 
	 *  Is called when the Agent is done if ResetOneDone is false.
	 *  The agent will remain done.
	 *  You can use this method to remove the agent from the scene. 
	*/
    public virtual void AgentOnDone()
    {

    }

    /// Defines agent-specific reset logic
    /** Must be implemented in agent-specific child class. 
	 *  Is called when the academy is done.  
	 *  Is called when the Agent is done if ResetOneDone is true.
	*/
    public virtual void AgentReset()
    {

    }

    /// Do not modify : Is used by the brain to reset the agent.
    public void Reset()
    {
        memory = new float[brain.brainParameters.memorySize];
        stepCounter = 0;
        AgentReset();
    }

    /// Do not modify : Is used by the brain to collect rewards.
    public float CollectReward()
    {
        return reward;
    }

    public void SetCumulativeReward()
    {
        CumulativeReward += reward;
        //Debug.Log(reward);
    }

    /// Do not modify : Is used by the brain to collect done.
    public bool CollectDone()
    {
        return done;
    }

    /// Do not modify : Is used by the brain give new action to the agent.
    public void UpdateAction(float[] a)
    {
        agentStoredAction = a;
    }

    /// Do not modify : Is used by the brain to make the agent perform a step.
    public void Step()
    {
        AgentStep(agentStoredAction);
        stepCounter += 1;
        if ((stepCounter > maxStep) && (maxStep > 0))
        {
            done = true;
        }
    }

    /// Do not modify : Is used by the brain to reset the Reward.
    public void ResetReward()
    {
        reward = 0;
        CumulativeReward = 0f;
    }

}
