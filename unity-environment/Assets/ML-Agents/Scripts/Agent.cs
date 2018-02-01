using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class AgentInfo
{
    public List<float> vectorObservation;
    public List<float> stakedVectorObservation;
    public List<Texture2D> visualObservations;
    public float[] memories;
    public string textObservation;
    public float[] StoredVectorActions;
    public string StoredTextActions;

    public float reward;
    public bool done;
    public bool maxStepReached;
    public int id;
}

public class AgentAction
{
    public float[] vectorActions;
    public string textActions;
    public float[] memories;
    public float value;
}


[HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Agents-Editor-Interface.md#agent")]
/** Generic functions for parent Agent class.
 * Contains all logic for Brain-Agent communication and Agent-Environment 
 * interaction.
 */
public abstract class Agent : MonoBehaviour
{
    [Tooltip("The brain to register this agent to. Can be dragged into the inspector using the Editor.")]
    /**<  \brief  The brain that will control this agent. */
    /**< Use the inspector to drag the desired brain gameObject into
	 * the Brain field */
    public Brain brain;

    [Tooltip("A list of Cameras which will be used to generate observations.")]
    /**<  \brief  The list of the cameras the Agent uses as observations. */
    /**< These cameras will be used to generate the observations */
    public List<Camera> agentCameras;

    [Tooltip("The per-agent maximum number of steps.")]
    /**<  \brief  The number of steps the agent takes before being done. */
    /**< If set to 0, the agent can only be set to done via a script.
    * If set to any positive integer, the agent will be set to done after that
    * many steps each episode. */
    public int maxStep;

    [Tooltip("If checked, the agent will reset on done. Else, AgentOnDone() will be called.")]
    /**<  \brief Determines the behaviour of the Agent when done.*/
    /**< If true, the agent will reset when done. 
	 * If not, the agent will remain done, and no longer take actions.*/
    public bool resetOnDone = true;


    private AgentInfo _info = new AgentInfo();
    private AgentAction _action = new AgentAction();



    /**< \brief Describes the reward for the given step of the agent.*/
    /**< It is reset to 0 at the beginning of every step. 
    * Modify in AgentStep(). 
    * Should be set to positive to reinforcement desired behavior, and
    * set to a negative value to punish undesireable behavior.
    * Additionally, the magnitude of the reward should not exceed 1.0 */
    [HideInInspector]
    public float reward;


    //TODO : Will be required for event based simulation
    [HideInInspector]
    public bool requestAction;
    [HideInInspector]
    public bool requestDecision;

    /**< \brief Whether or not the agent is done*/
    /**< Set to true when the agent has acted in some way which ends the 
     * episode for the given agent. */
    [HideInInspector]
    public bool done;

    /**< \brief Whether or not the max step is reached*/
    [HideInInspector]
    public bool maxStepReached;

    /**< \brief The current value estimate of the agent */
    /**<  When using an External brain, you can pass value estimates to the
     * agent at every step using env.Step(actions, values).
     * If AgentMonitor is attached to the Agent, this value will be displayed.*/
    [HideInInspector]
    public float value;

    /**< \brief Do not modify: This keeps track of the cumulative reward.*/
    [HideInInspector]
    public float CumulativeReward;

    /**< \brief Do not modify: This keeps track of the number of steps taken by
     * the agent each episode.*/
    [HideInInspector]
    public int stepCounter;


    /**< \brief Do not modify : This is the unique Identifier each agent 
     * receives at initialization. It is used by the brain to identify
     * the agent.*/
    [HideInInspector]
    public int id;

    private void OnEnable()
    {
        _InitializeAgent();
    }

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

    void _DisableAgent()
    {
        Academy aca = Object.FindObjectOfType<Academy>() as Academy;
        if (aca != null)
            aca.UnRegisterAgent(this);
    }

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
        _info.memories = new float[brain.brainParameters.memorySize];
        if (brain.brainParameters.stateSpaceType == StateType.continuous)
        {
            _info.vectorObservation = new List<float>(brain.brainParameters.stateSize);
            _info.stakedVectorObservation = new List<float>(brain.brainParameters.stateSize * brain.brainParameters.stackedStates);
            _info.stakedVectorObservation.AddRange(new float[brain.brainParameters.stateSize * brain.brainParameters.stackedStates]);
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

    public void SendStateToBrain()
    {
        SetCumulativeReward();
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
				throw new UnityAgentsException(string.Format(@"Vector Observation size mismatch between discreete agent {0} and
                    brain {1}. Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.gameObject.name,
					1, _info.vectorObservation.Count));
            }
            _info.stakedVectorObservation.RemoveRange(0, 1);
            _info.stakedVectorObservation.AddRange(_info.vectorObservation);
        }
        _info.visualObservations.Clear();
        if (brain.brainParameters.cameraResolutions.Length > agentCameras.Count)
        {
            throw new UnityAgentsException(string.Format(@"Not enough cameras for agent {0} : 
                Bain {1} expecting at least {2} cameras but only {3} were present.",
                gameObject.name, brain.gameObject.name,
                brain.brainParameters.cameraResolutions.Length, agentCameras.Count));
        }
        for (int i = 0; i < brain.brainParameters.cameraResolutions.Length; i++)
        {
            _info.visualObservations.Add(ObservationToTexture(
                agentCameras[i],
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

    internal void AddVectorObs(float v)
    {
        _info.vectorObservation.Add(v);
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
        AgentReset();
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
    public void UpdateMemoriesAction(float[] v)
    {
        _action.memories = v;
    }
    public void UpdateValueAction(float v)
    {
        _action.value = v;
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
            requestAction = false;
            AgentAction(_action.vectorActions);
        }

        stepCounter += 1;
        if ((stepCounter > maxStep) && (maxStep > 0))
        {
            maxStepReached = true;
            // This is temporary :
            done = true;
            maxStepReached = true;
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
