using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Agent info.The agent will send an instance of this class to the brain.
/// </summary>
public struct AgentInfo
{
    public List<float> vectorObservation;
    public List<float> stackedVectorObservation;
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
public struct AgentAction
{
    public float[] vectorActions;
    public string textActions;
    public List<float> memories;
}

/// <summary>
/// Agent parameters. Reflect the user's settings for the agents of the inspector.
/// </summary>
[System.Serializable]
public class AgentParameters
{
    public List<Camera> agentCameras = new List<Camera>();
    public int maxStep;
    public bool resetOnDone = true;
    public bool onDemandDecision;
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
    /// <summary>
    /// The brain that will control this agent. 
    /// Use the inspector to drag the desired brain gameObject into
	/// the Brain field.
    ///</summary>
    [HideInInspector]
    public Brain brain;

    /// <summary>
    /// The info. This is the placeholder for the information the agent will send
    /// to the brain.
    /// </summary>
    private AgentInfo _info;
    /// <summary>
    /// The action. This is the placeholder for the actions the agent will receive.
    /// </summary>
    private AgentAction _action;

    /// <summary>
    /// The reward. Describes the reward for the given step of the agent.
    /// It is reset to 0 at the beginning of every step. 
    /// Modify in AgentStep(). 
    /// Should be set to positive to reinforcement desired behavior, and
    /// set to a negative value to punish undesireable behavior.
    /// Additionally, the magnitude of the reward should not exceed 1.0
    ///</summary>
    private float reward;

    /// Whether or not the agent requests an action
    private bool requestAction;

    /// Whether or not the agent requests a decision
    private bool requestDecision;

    /// <summary> 
    /// Whether or not the agent is done
    /// Set to true when the agent has acted in some way which ends the 
    /// episode for the given agent. 
    ///</summary>
    private bool done;

    /// Whether or not the max step is reached
    private bool maxStepReached;

    /// Do not modify: This keeps track of the cumulative reward.
    private float cumulativeReward;

    /// This keeps track of the number of steps taken by the agent each episode.
    [HideInInspector]
    public int stepCounter;

    private bool hasAlreadyReset;
    private bool terminate;

    [HideInInspector]
    public AgentParameters agentParameters;

    /// <summary> This is the unique Identifier each agent 
    /// receives at initialization. It is used by the brain to identify
    /// the agent.
    /// </summary>
    private int id;

    /// <summary>
	/// Unity method called when the agent is instantiated or set to active.
    /// </summary>
    private void OnEnable()
    {
        id = gameObject.GetInstanceID();
        Academy aca = Object.FindObjectOfType<Academy>() as Academy;
        _InitializeAgent(aca);
    }

    /// <summary>
    /// Is called when the agent is initialized. 
    /// </summary>
    void _InitializeAgent(Academy aca)
    {
        _info = new AgentInfo();
        _action = new AgentAction();

        if (aca == null)
            throw new UnityAgentsException("No Academy Component could be" +
                                           "found in the scene.");
        aca.AgentSetStatus += SetStatus;
        aca.AgentResetIfDone += ResetIfDone;
        aca.AgentSendState += SendState;
        aca.AgentAct += _AgentStep;
        aca.AgentForceReset += _AgentReset;

        if (brain != null)
        {
            ResetState();
        }
        else
        {
            Debug.Log(
                string.Format("The Agent component attached to the " +
                            "GameObject {0} was initialized without a brain."
                              , gameObject.name));
        }

        InitializeAgent();
    }

    /// <summary>
    /// Is called when the agent is disabled. 
    /// </summary>
    void _DisableAgent(Academy aca)
    {
        if (aca != null)
        {
            aca.AgentSetStatus -= SetStatus;
            aca.AgentResetIfDone -= ResetIfDone;
            aca.AgentSendState -= SendState;
            aca.AgentAct -= _AgentStep;
            aca.AgentForceReset -= _AgentReset;
        }
    }

    /// <summary>
    /// Gets called when the agent is destroyed or is set inactive.
    /// </summary>
    void OnDisable()
    {
        Academy aca = Object.FindObjectOfType<Academy>() as Academy;
        _DisableAgent(aca);
    }

    /// <summary>
    /// When GiveBrain is called, the agent unsubscribes from its 
    /// previous brain and subscribes to the one passed in argument.
    /// Use this method to provide a brain to the agent via script. 
	///<param name= "b" >The Brain the agent will subscribe to.</param>
    /// <summary>
    public void GiveBrain(Brain b)
    {
        brain = b;
        ResetState();

    }
    /// <summary>
    /// Resets the reward of the agent
    /// </summary>
    public void ResetReward()
    {
        reward = 0f;
        if (done)
        {
            cumulativeReward = 0f;
        }
    }
    /// <summary>
    /// Use this method to overrite the current reward of the agent.
    /// </summary>
    /// <param name="newValue">The new value of the reward</param>
    public void SetReward(float newValue)
    {
        cumulativeReward += newValue - reward;
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
        cumulativeReward += increment;
    }
    /// <summary>
    /// Gets the reward of the agent.
    /// </summary>
    /// <returns>The reward.</returns>
    public float GetReward()
    {
        return reward;
    }
    /// <summary>
    /// Gets the cumulative reward.
    /// </summary>
    public float GetCumulativeReward()
    {
        return cumulativeReward;
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
        RequestAction();
    }
    /// <summary>
    /// Is called then the agent must perform a new action.
    /// </summary>
	public void RequestAction()
    {
        requestAction = true;
    }
    /// <summary>
    /// Indicates if the agent has reached his maximum number of steps.
    /// </summary>
    /// <returns><c>true</c>, if max step reached was reached,
    ///  <c>false</c> otherwise.</returns>
    public bool IsMaxStepReached()
    {
        return maxStepReached;
    }
    /// <summary>
    /// Indicates if the agent is done
    /// </summary>
    /// <returns><c>true</c>, if the agent is done,
    ///  <c>false</c> otherwise.</returns>
    public bool IsDone()
    {
        return done;
    }

    /// <summary>
    /// Resets the info and action fields of the agent. Is called when the agent
    /// resets or changes brain.
    /// </summary>
    private void ResetState()
    {
        if (brain == null)
            return;

        BrainParameters param = brain.brainParameters;
        if (param.vectorActionSpaceType == StateType.continuous)
        {
            _action.vectorActions = new float[param.vectorActionSize];
            _info.StoredVectorActions = new float[param.vectorActionSize];
        }
        else
        {
            _action.vectorActions = new float[1];
            _info.StoredVectorActions = new float[1];
        }
        _action.textActions = "";
        _info.memories = new List<float>();
        _action.memories = new List<float>();
        if (param.vectorObservationSpaceType == StateType.continuous)
        {
            _info.vectorObservation =
                new List<float>(param.vectorObservationSize);
            _info.stackedVectorObservation =
                new List<float>(param.vectorObservationSize
                        * brain.brainParameters.numStackedVectorObservations);
            _info.stackedVectorObservation.AddRange(
                new float[param.vectorObservationSize
                          * param.numStackedVectorObservations]);
        }
        else
        {
            _info.vectorObservation = new List<float>(1);
            _info.stackedVectorObservation =
                new List<float>(param.numStackedVectorObservations);
            _info.stackedVectorObservation.AddRange(
                new float[param.numStackedVectorObservations]);
        }
        _info.visualObservations = new List<Texture2D>();
    }


    /// <summary>
    /// Initialize the agent with this method
    /// Must be implemented in agent-specific child class.
	/// This method called only once when the agent is enabled.
    /// </summary>
    public virtual void InitializeAgent()
    {

    }

    /// <summary>
    /// Sends the state to brain.
    /// </summary>
    public void SendStateToBrain()
    {
        if (brain == null)
            return;
        _info.memories = _action.memories;
        _info.StoredVectorActions = _action.vectorActions;
        _info.StoredTextActions = _action.textActions;
        _info.vectorObservation.Clear();
        CollectObservations();

        BrainParameters param = brain.brainParameters;
        if (param.vectorObservationSpaceType == StateType.continuous)
        {
            if (_info.vectorObservation.Count != param.vectorObservationSize)
            {
                throw new UnityAgentsException(string.Format(
                    "Vector Observation size mismatch between continuous " +
                    "agent {0} and brain {1}. " +
                    "Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.gameObject.name,
                    brain.brainParameters.vectorObservationSize,
                    _info.vectorObservation.Count));
            }
            _info.stackedVectorObservation.RemoveRange(
                0, param.vectorObservationSize);
            _info.stackedVectorObservation.AddRange(_info.vectorObservation);
        }
        else
        {
            if (_info.vectorObservation.Count != 1)
            {
                throw new UnityAgentsException(string.Format(
                    "Vector Observation size mismatch between discrete agent" +
                    " {0} and brain {1}. Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.gameObject.name,
                    1, _info.vectorObservation.Count));
            }
            _info.stackedVectorObservation.RemoveRange(0, 1);
            _info.stackedVectorObservation.AddRange(_info.vectorObservation);
        }
        _info.visualObservations.Clear();
        if (param.cameraResolutions.Length > agentParameters.agentCameras.Count)
        {
            throw new UnityAgentsException(string.Format(
                "Not enough cameras for agent {0} : Bain {1} expecting at " +
                "least {2} cameras but only {3} were present.",
                gameObject.name, brain.gameObject.name,
                brain.brainParameters.cameraResolutions.Length,
                agentParameters.agentCameras.Count));
        }
        for (int i = 0; i < brain.brainParameters.cameraResolutions.Length; i++)
        {
            _info.visualObservations.Add(ObservationToTexture(
                agentParameters.agentCameras[i],
                param.cameraResolutions[i].width,
                param.cameraResolutions[i].height));
        }

        _info.reward = reward;
        _info.done = done;
        _info.maxStepReached = maxStepReached;
        _info.id = id;


        brain.SendState(this, _info);
        _info.textObservation = "";
    }
    /// <summary>
    /// Collects the observations. Must be implemented by the developer.
    /// </summary>
    public virtual void CollectObservations()
    {

    }

    /// <summary>
    /// Adds a vector observation. 
    /// Note that the number of vector observation to add
    /// must be the same at each CollectObservations call.
    /// </summary>
    /// <param name="observation">The float value to add to 
    /// the vector observation.</param>
    internal void AddVectorObs(float observation)
    {
        _info.vectorObservation.Add(observation);
    }

    internal void SetTextObs(object s)
    {
        _info.textObservation = s.ToString();
    }

    /// <summary>
    /// Defines agent-specific behavior at every step depending on the action.
    /// Must be implemented in agent-specific child class.
    /// Note: If your state is discrete, you need to convert your 
    /// state into a list of float with length 1.
    /// </summary>
    /// <param name="action">The action the agent receives 
    /// from the brain.</param>
    public virtual void AgentAction(float[] action)
    {

    }

    /// <summary>
    /// Defines agent-specific behaviour when done
    /// Must be implemented in agent-specific child class. 
    /// Is called when the Agent is done if ResetOneDone is false.
    /// The agent will remain done.
    /// You can use this method to remove the agent from the scene. 
    /// </summary>
    public virtual void AgentOnDone()
    {

    }


    /// <summary>
    /// Defines agent-specific reset logic
    /// Must be implemented in agent-specific child class. 
    /// Is called when the academy is done.  
    /// Is called when the Agent is done if ResetOneDone is true.
    /// </summary>
    public virtual void AgentReset()
    {

    }

    /// <summary>
    /// Is called when the agent resets.
    /// </summary>
    public void _AgentReset()
    {
        ResetState();
        stepCounter = 0;
        AgentReset();
    }

    /// Is used by the brain give new action to the agent.
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
    public void UpdateTextAction(string t)
    {
        _action.textActions = t;
    }

    /// <summary>
    /// Sets the status of the agent.
    /// </summary>
    /// <param name="acaMaxStep">If set to <c>true</c> 
    /// The agent must set maxStepReached.</param>
    /// <param name="acaDone">If set to <c>true</c> 
    /// The agent must set done.</param>
    private void SetStatus(bool acaMaxStep, bool acaDone, int acaStepCounter)
    {
        if (acaDone)
            acaStepCounter = 0;
        MakeRequests(acaStepCounter);
        if (acaMaxStep)
            maxStepReached = true;
        if (acaDone)
        {
            Done();
            hasAlreadyReset = false;
            // If the Academy needs to reset, the agent should reset 
            // even if it reseted recently.
        }
    }

    /// <summary>
    /// Signals the agent that it must reset if its done flag is set to true.
    /// </summary>
    private void ResetIfDone()
    {
        // If an agent is done, then it will also 
        // request for a decision and an action
        if (IsDone())
        {
            if (agentParameters.resetOnDone)
            {
                if (agentParameters.onDemandDecision)
                {
                    if (!hasAlreadyReset)
                    {
                        //If event based, the agent can reset as soon
                        // as it is done
                        _AgentReset();
                        hasAlreadyReset = true;
                    }
                }
                else if (requestDecision)
                {
                    // If not event based, the agent must wait to request a
                    // decsion before reseting to keep multiple agents in sync.
                    _AgentReset();
                }
            }
            else
            {
                terminate = true;
                RequestDecision();
            }
        }
    }

    /// <summary>
    /// Signals the agent that it must sent its decision to the brain.
    /// </summary>
    private void SendState()
    {
        if (requestDecision)
        {
            SendStateToBrain();
            ResetReward();
            done = false;
            maxStepReached = false;
            requestDecision = false;

            hasAlreadyReset = false;
        }
    }

    ///  Is used by the brain to make the agent perform a step.
    private void _AgentStep()
    {

        if (terminate)
        {
            terminate = false;
            ResetReward();
            done = false;
            maxStepReached = false;
            requestDecision = false;
            requestAction = false;

            hasAlreadyReset = false;
            OnDisable();
            AgentOnDone();
        }


        if ((requestAction) && (brain != null))
        {
            requestAction = false;
            AgentAction(_action.vectorActions);
        }

        if ((stepCounter >= agentParameters.maxStep)
            && (agentParameters.maxStep > 0))
        {
            maxStepReached = true;
            Done();
        }
        stepCounter += 1;
    }
    /// <summary>
    /// Is called after every step, contains the logic to decide if the agent
    /// will request a decision at the next step.
    /// </summary>
    private void MakeRequests(int acaStepCounter)
    {
        agentParameters.numberOfActionsBetweenDecisions =
    Mathf.Max(agentParameters.numberOfActionsBetweenDecisions, 1);
        if (!agentParameters.onDemandDecision)
        {
            RequestAction();
            if (acaStepCounter %
                agentParameters.numberOfActionsBetweenDecisions == 0)
            {
                RequestDecision();
            }
        }
    }

    /** Contains logic for coverting a camera component into a Texture2D. */
    public Texture2D ObservationToTexture(Camera cam, int width, int height)
    {
        Rect oldRec = cam.rect;
        cam.rect = new Rect(0f, 0f, 1f, 1f);
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;

        var tempRT =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite);
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);

        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = tempRT;
        cam.targetTexture = tempRT;

        cam.Render();

        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();
        cam.targetTexture = prevCameraRT;
        cam.rect = oldRec;
        RenderTexture.active = prevActiveRT;
        RenderTexture.ReleaseTemporary(tempRT);
        return tex;

    }


}
