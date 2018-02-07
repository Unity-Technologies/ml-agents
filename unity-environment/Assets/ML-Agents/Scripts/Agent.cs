using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Agent info.The agent will send an instance of this class to the brain.
/// </summary>
public class AgentInfo
{
    public List<float> vectorObservation;
    public List<float> stakedVectorObservation;
    public List<Texture2D> visualObservations;
    public List<float> memories;
    public string textObservation;
    public float[] StoredVectorActions;
    public string StoredTextActions;

    public float reward;
    public bool done;
    public bool maxStepReached;
    public int id;
}

/// <summary>
/// Agent action. The brain will send an instance of this class to the agent
///  when taking a decision.
/// </summary>
public class AgentAction
{
    public float[] vectorActions;
    public string textActions;
    public List<float> memories;
    public float valueEstimate;
}

[System.Serializable]
public class AgentParameters
{
    public List<Camera> agentCameras;
    public int maxStep;
    public bool resetOnDone = true;
	public bool eventBased;
	public int numberOfStepsBetweenActions;
	public int numberOfActionsBetweenDecisions;
}


[HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Agents-Editor-Interface.md#agent")]
/** Generic functions for parent Agent class.
 * Contains all logic for Brain-Agent communication and Agent-Environment 
 * interaction.
 */
[System.Serializable]
public abstract class Agent : MonoBehaviour
{
    /**<  \brief  The brain that will control this agent. */
    /**< Use the inspector to drag the desired brain gameObject into
	 * the Brain field */
    [HideInInspector]
    //[SerializeField]
    public Brain brain;

    /**<  \brief  The list of the cameras the Agent uses as observations. */
    /**< These cameras will be used to generate the observations */
    //[HideInInspector]
    //[SerializeField]
    //public List<Camera> agentCameras;

	/**<  \brief  The number of steps the agent takes before being done. */
	/**< If set to 0, the agent can only be set to done via a script.
    * If set to any positive integer, the agent will be set to done after that
    * many steps each episode. */
	//[HideInInspector]
    //[SerializeField]
    //public int maxStep;

    /**<  \brief Determines the behaviour of the Agent when done.*/
    /**< If true, the agent will reset when done. 
	 * If not, the agent will remain done, and no longer take actions.*/
    //[HideInInspector]
    //[SerializeField]
    //public bool resetOnDone = true;

    /// <summary>
    /// The info. This is the placeholder for the information the agent will send
    /// to the brain.
    /// </summary>
    private AgentInfo _info = new AgentInfo();
    /// <summary>
    /// The action. This is the placeholder for the actions the agent will receive.
    /// </summary>
    private AgentAction _action = new AgentAction();

    /**< \brief Describes the reward for the given step of the agent.*/
    /**< It is reset to 0 at the beginning of every step. 
    * Modify in AgentStep(). 
    * Should be set to positive to reinforcement desired behavior, and
    * set to a negative value to punish undesireable behavior.
    * Additionally, the magnitude of the reward should not exceed 1.0 */
    private float reward;

    /**< \brief Whether or not the agent is requests an action*/
    private bool requestAction;

    /**< \brief Whether or not the agent is requests a decision*/
    private bool requestDecision;

    /**< \brief Whether or not the agent is done*/
    /**< Set to true when the agent has acted in some way which ends the 
     * episode for the given agent. */
    private bool done;

    /**< \brief Whether or not the max step is reached*/
    private bool maxStepReached;

    /**< \brief Do not modify: This keeps track of the cumulative reward.*/
    private float CumulativeReward;

    /**< \brief Do not modify: This keeps track of the number of steps taken by
     * the agent each episode.*/
    private int stepCounter;

	private int stepsSinceAction;
	private int actionsSinceDecision;

    //[HideInInspector]
    //[SerializeField]
    //public bool eventBased;
    //[HideInInspector]
    //[SerializeField]
    //public int numberOfStepsBetweenActions;
    //[HideInInspector]
    //[SerializeField]
    //public int numberOfActionsBetweenDecisions;
    //[SerializeField]
    [HideInInspector]
    public AgentParameters agentParameters;

    /**< \brief This is the unique Identifier each agent 
     * receives at initialization. It is used by the brain to identify
     * the agent.*/
    private int id;

    /// <summary>
    /// Unity method called when the agent is istanciated or set to active.
    /// </summary>
    private void OnEnable()
    {
        _InitializeAgent();
    }

    /// <summary>
    /// Is called when the agent is initialized. 
    /// </summary>
    void _InitializeAgent()
    {
        id = gameObject.GetInstanceID();
        Academy aca = Object.FindObjectOfType<Academy>() as Academy;
        if (aca == null)
            Debug.Log("No Academy Object could be found");
        aca.RegisterAgent(this);
        if (brain != null)
        {
            ResetState();
        }
        InitializeAgent();
    }

    /// <summary>
    /// Is called when the agent is disabled. 
    /// </summary>
    void _DisableAgent()
    {
        Academy aca = Object.FindObjectOfType<Academy>() as Academy;
        if (aca != null)
            aca.UnRegisterAgent(this);
    }

    /// <summary>
    /// Unity Method. Gets called when the agent is destroyed or is set inactive.
    /// </summary>
    void OnDisable()
    {
        _DisableAgent();
    }

    /// When GiveBrain is called, the agent unsubscribes from its 
    /// previous brain and subscribes to the one passed in argument.
    /** Use this method to provide a brain to the agent via script. 
	 * Do not modify brain directly.
	@param b The Brain component the agent will subscribe to.*/
    public void GiveBrain(Brain b)
    {
        brain = b;
        ResetState();

    }

    /// <summary>
    /// Use this method to overrite the current reward of the agent.
    /// </summary>
    /// <param name="newValue">The new value of the reward</param>
    public void SetReward(float newValue)
    {
        reward = newValue;
    }
    /// <summary>
    /// Use this method to increment the current reward of the agent.
    /// </summary>
    /// <param name="increment">The value by which the reward will
    /// be incremented</param>
    public void AddReward(float increment)
    {
        reward += increment;
    }
    /// <summary>
    /// Gets the reward of the agent.
    /// </summary>
    /// <returns>The reward.</returns>
    public float GetReward(){
        return reward;
    }
    /// <summary>
    /// Gets the value estimate of the agent.
    /// </summary>
    /// <returns>The value estimate.</returns>
    public float GetValue(){
        return _action.valueEstimate;
    }
    /// <summary>
    /// Is called when the agent reaches the maximum number of steps.
    /// </summary>
    public void MaxStepReached()
    {
        maxStepReached =true;
		// When the maxStepReached flag is set, the done flag must also be set.
		Done();
    }
    /// <summary>
    /// Is called then the agent is done. Either game-over, victory or timeout.
    /// </summary>
    public void Done()
    {
        done = true;
    }
    /// <summary>
    /// Is called when the agent must request the brain for a new decision.
    /// </summary>
	public void RequestDecision()
	{
        requestDecision = true;
        // When the agent requests  decision, it must also request an action.
		RequestAction();
	}
    /// <summary>
    /// Is called then the agent must perform a new action.
    /// </summary>
	public void RequestAction()
	{
		requestAction = true;
	}
    public void _ClearMaxStepReached(){
        maxStepReached = false;
    }
    public void _ClearDone(){
        done = false;
    }
    public void _ClearRequestDecision(){
        requestDecision = false;
    }
    public void _ClearRequestAction(){
        requestAction = false;
    }
    /// <summary>
    /// Indicates if the agent has reached his maximum number of steps.
    /// </summary>
    /// <returns><c>true</c>, if max step reached was reached,
    ///  <c>false</c> otherwise.</returns>
    public bool IsMaxStepReached(){
        return maxStepReached;
    }
    /// <summary>
    /// Indicates if the agent is done
    /// </summary>
    /// <returns><c>true</c>, if the agent is done,
    ///  <c>false</c> otherwise.</returns>
    public bool IsDone(){
        return done;
    }
    /// <summary>
    /// Indicates if the agent has requested a decision
    /// </summary>
    /// <returns><c>true</c>, if the agent has requested a decision,
    ///  <c>false</c> otherwise.</returns>
    public bool HasRequestedDecision(){
        return requestDecision;
    }
    /// <summary>
    /// Indicates if the agent has requested an action
    /// </summary>
    /// <returns><c>true</c>, if the agent has requested an action,
    ///  <c>false</c> otherwise.</returns>
    public bool HasRequestedAction(){
        return requestAction;
    }

    /// <summary>
    /// Resets the info and action fields of the agent. Is called when the agent
    /// resets or changes brain.
    /// </summary>
    internal void ResetState()
    {
		if (brain.brainParameters.actionSpaceType == StateType.continuous)
        {
            _action.vectorActions = new float[brain.brainParameters.actionSize];
            _info.StoredVectorActions = new float[brain.brainParameters.actionSize];
        }
        else
        {
            _action.vectorActions = new float[1];
            _info.StoredVectorActions = new float[1];
        }
        _action.textActions = "";
        _info.memories = new List<float>();
        _action.memories = new List<float>();
        if (brain.brainParameters.stateSpaceType == StateType.continuous)
        {
            _info.vectorObservation = new List<float>(brain.brainParameters.stateSize);
            _info.stakedVectorObservation = new List<float>(brain.brainParameters.stateSize 
                                                            * brain.brainParameters.stackedStates);
            _info.stakedVectorObservation.AddRange(new float[brain.brainParameters.stateSize 
                                                             * brain.brainParameters.stackedStates]);
        }
        else
        {
            _info.vectorObservation = new List<float>(1);
            _info.stakedVectorObservation = new List<float>(brain.brainParameters.stackedStates);
            _info.stakedVectorObservation.AddRange(new float[brain.brainParameters.stackedStates]);
        }
        _info.visualObservations = new List<Texture2D>();
    }


    /// Initialize the agent with this method
    /** Must be implemented in agent-specific child class.
	 *  This method called only once when the agent is enabled.
	*/
    public virtual void InitializeAgent()
    {

    }

    /// <summary>
    /// Sends the state to brain.
    /// </summary>
    public void SendStateToBrain()
    {
        actionsSinceDecision = 0;
        SetCumulativeReward();
        _info.memories = _action.memories;
        _info.StoredVectorActions = _action.vectorActions;
        _info.StoredTextActions = _action.textActions;
        _info.vectorObservation.Clear();
        _info.textObservation = "";
        CollectObservations();

        if (brain.brainParameters.stateSpaceType == StateType.continuous)
        {
            if (_info.vectorObservation.Count != brain.brainParameters.stateSize)
            {
                throw new UnityAgentsException(string.Format(@"Vector Observation size mismatch between continuous agent {0} and
                    brain {1}. Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.gameObject.name,
                    brain.brainParameters.stateSize, _info.vectorObservation.Count));
            }
            _info.stakedVectorObservation.RemoveRange(0, brain.brainParameters.stateSize);
            _info.stakedVectorObservation.AddRange(_info.vectorObservation);
        }
        else
        {
            if (_info.vectorObservation.Count != 1)
            {
                throw new UnityAgentsException(string.Format(@"Vector Observation size mismatch between discrete agent {0} and
                    brain {1}. Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.gameObject.name,
                    1, _info.vectorObservation.Count));
            }
            _info.stakedVectorObservation.RemoveRange(0, 1);
            _info.stakedVectorObservation.AddRange(_info.vectorObservation);
        }
        _info.visualObservations.Clear();
        if (brain.brainParameters.cameraResolutions.Length > agentParameters.agentCameras.Count)
        {
            throw new UnityAgentsException(string.Format(@"Not enough cameras for agent {0} : 
                Bain {1} expecting at least {2} cameras but only {3} were present.",
                gameObject.name, brain.gameObject.name,
                brain.brainParameters.cameraResolutions.Length, agentParameters.agentCameras.Count));
        }
        for (int i = 0; i < brain.brainParameters.cameraResolutions.Length; i++)
        {
            _info.visualObservations.Add(ObservationToTexture(
                agentParameters.agentCameras[i],
                brain.brainParameters.cameraResolutions[i].width,
                brain.brainParameters.cameraResolutions[i].height));
        }

        _info.reward = reward;
        _info.done = done;
        _info.maxStepReached = maxStepReached;
        _info.id = id;


        brain.SendState(this, _info);

    }

    public virtual void CollectObservations()
    {
        // The develloper will use AddVectorObs()
    }

    /// <summary>
    /// Adds a vector observation. Note that the number of vector observation to add
    /// must be the same at each CollectObservations call.
    /// </summary>
    /// <param name="observation">The float value to add to the vector observation.</param>
    internal void AddVectorObs(float observation)
    {
        _info.vectorObservation.Add(observation);
    }
    internal void AddTextObs(string s)
    {
        _info.textObservation += s;
    }


    /// Defines agent-specific behavior at every step depending on the action.
    /** Must be implemented in agent-specific child class.
	 *  Note: If your state is discrete, you need to convert your 
	 *  state into a list of float with length 1.
	 *  @param action The action the agent receives from the brain. 
	*/
    public virtual void AgentAction(float[] action)
    {
        //Is it needed to pass the action since the developer has access directly to action ?

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
    public void _AgentReset()
    {
        ResetState();
        stepCounter = 0;
        //stepsSinceAction = 0;
        //actionsSinceDecision = 0;
        AgentReset();
    }

    public void _resetCounters(){
		stepCounter = 0;
		stepsSinceAction = 0;
		actionsSinceDecision = 0;
	}


    public void SetCumulativeReward()
    {
        if (!done)
        {
            CumulativeReward += reward;
        }
        else
        {
            CumulativeReward = 0f;
        }
    }


    /// Do not modify : Is used by the brain give new action to the agent.
    public void UpdateAction(AgentAction action)
    {
        _action = action;
    }
    public void UpdateVectorAction(float[] v)
    {
        _action.vectorActions = v;
    }
    public void UpdateMemoriesAction(List<float> v)
    {
        _action.memories = v;
    }
    public void UpdateValueAction(float v)
    {
        _action.valueEstimate = v;
    }
    public void UpdatevTextAction(string t)
    {
        _action.textActions = t;
    }


    /// Do not modify : Is used by the brain to make the agent perform a step.
    public void _AgentStep()
    {
        
        //AgentStep(_action.vectorActions);

        if (requestAction)
        {
            _ClearRequestAction();
            AgentAction(_action.vectorActions);
            stepsSinceAction = 0;
            actionsSinceDecision += 1;
        }
        stepsSinceAction += 1;

        if ((stepCounter > agentParameters.maxStep) && (agentParameters.maxStep > 0))
        {
            MaxStepReached();
        }
        stepCounter += 1;
        MakeRequests();
    }
    /// <summary>
    /// Is called after every step, contains the logic to decide if the agent
    /// will request a decision at the next step.
    /// </summary>
    public void MakeRequests(){
        if(!agentParameters.eventBased){
            if (stepsSinceAction >= agentParameters.numberOfStepsBetweenActions){
                RequestAction();
            }
            if (actionsSinceDecision >= agentParameters.numberOfActionsBetweenDecisions){
                RequestDecision();
            }
        }
    }



    /** Contains logic for coverting a camera component into a Texture2D. */
    public Texture2D ObservationToTexture(Camera camera, int width, int height)
    {
        Camera cam = camera;
        Rect oldRec = camera.rect;
        cam.rect = new Rect(0f, 0f, 1f, 1f);
        bool supportsAntialiasing = false;
        bool needsRescale = false;
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;
        var antiAliasing = (supportsAntialiasing) ? Mathf.Max(1, QualitySettings.antiAliasing) : 1;

        var finalRT =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var renderRT = (!needsRescale) ? finalRT :
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);

        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = renderRT;
        cam.targetTexture = renderRT;

        cam.Render();

        if (needsRescale)
        {
            RenderTexture.active = finalRT;
            Graphics.Blit(renderRT, finalRT);
            RenderTexture.ReleaseTemporary(renderRT);
        }

        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();
        cam.targetTexture = prevCameraRT;
        cam.rect = oldRec;
        RenderTexture.active = prevActiveRT;
        RenderTexture.ReleaseTemporary(finalRT);
        return tex;
    }


}
