using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*! \mainpage ML-Agents Index Page
 * Welcome to Unity Machine Learning Agents documentation.
 */


[System.Serializable]
public class ScreenConfiguration
{
    [Tooltip("Width of the environment window in pixels.")]
    public int width;
    [Tooltip("Height of the environment window in pixels")]
    public int height;
    [Tooltip("Rendering quality of environment. (Higher is better quality)")]
    [Range(0, 5)]
    public int qualityLevel;
    [Tooltip("Speed at which environment is run. (Higher is faster)")]
    [Range(1f, 100f)]
    public float timeScale;
    [Tooltip("FPS engine attempts to maintain.")]
    public int targetFrameRate;

    public ScreenConfiguration(int w, int h, int q, float ts, int tf)
    {
        width = w;
        height = h;
        qualityLevel = q;
        timeScale = ts;
        targetFrameRate = tf;
    }
}


[HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Agents-Editor-Interface.md#academy")]
/** Create a child class to implement InitializeAcademy(), AcademyStep() 
 * and AcademyReset(). The child class script must be attached to an empty game
 * object in your scene, and there can only be one such object within the scene.
 */
public abstract class Academy : MonoBehaviour
{




    [SerializeField]
    [Tooltip("Total number of steps per episode. \n" +
             "0 corresponds to episodes without a maximum number of steps. \n" +
             "Once the step counter reaches maximum, the environment will reset.")]
    private int maxSteps;

    [SerializeField]
    [HideInInspector]
    public bool isInference = true;
    /**< \brief Do not modify : If true, the Academy will use inference 
     * settings. */
    private bool _isCurrentlyInference;
    [SerializeField]
    [Tooltip("The engine-level settings which correspond to rendering quality and engine speed during Training.")]
    private ScreenConfiguration trainingConfiguration = new ScreenConfiguration(80, 80, 1, 100.0f, -1);
    [SerializeField]
    [Tooltip("The engine-level settings which correspond to rendering quality and engine speed during Inference.")]
    private ScreenConfiguration inferenceConfiguration = new ScreenConfiguration(1280, 720, 5, 1.0f, 60);

    /**< \brief Contains a mapping from parameter names to float values. */
    /**< You can specify the Default Reset Parameters in the Inspector of the
     * Academy. You can modify these parameters when training with an External 
     * brain by passing a config dictionary at reset. Reference resetParameters
     * in your AcademyReset() or AcademyStep() to modify elements in your 
     * environment at reset time. */
    [SerializeField]
    [Tooltip("List of custom parameters that can be changed in the environment on reset.")]
    public ResetParameters resetParameters;

    private List<Brain> brains = new List<Brain>();

    private List<Agent> agents = new List<Agent>();

    private List<Agent> agentsTerminate = new List<Agent>();

    private List<Agent> agentsHasAlreadyReset = new List<Agent>();

    /**< \brief The done flag of the Academy. */
    /**< When set to true, the Academy will call AcademyReset() instead of 
    * AcademyStep() at step time.
    * If true, all agents done flags will be set to true.*/
    private bool done;

    /// <summary>
    /// The max step reached.
    /// </summary>
    private bool maxStepReached;


    /**< \brief Increments each time the environment is reset. */
    [HideInInspector]
    public int episodeCount;
    [HideInInspector]
    public int stepsSinceReset;

    ExternalCommand externalCommand;
    /**< \brief Do not modify : pointer to the communicator currently in 
     * use by the Academy. */
    public Communicator communicator;



    void Awake()
    {

        GetBrains(gameObject, brains);
        InitializeAcademy();

        communicator = new ExternalCommunicator(this);
        if (!communicator.CommunicatorHandShake())
        {
            communicator = null;
        }

        foreach (Brain brain in brains)
        {
            brain.InitializeBrain();
        }
        if (communicator != null)
        {
            communicator.InitializeCommunicator();
            communicator.UpdateCommand();

            externalCommand = communicator.GetCommand();
        }

        isInference = (communicator == null);
        _isCurrentlyInference = !isInference;
        done = true;
    }

    public void RegisterAgent(Agent a)
    {
        agents.Add(a);
    }

    public void UnRegisterAgent(Agent a)
    {
        if (agents.Contains(a))
            agents.Remove(a);
    }

    /// Environment specific initialization.
    /**
	* Implemented in environment-specific child class. 
	* This method is called once when the environment is loaded.
	*/
    public virtual void InitializeAcademy()
    {

    }


    private void ConfigureEngine()
    {
        if ((!isInference))
        {
            Screen.SetResolution(trainingConfiguration.width, trainingConfiguration.height, false);
            QualitySettings.SetQualityLevel(trainingConfiguration.qualityLevel, true);
            Time.timeScale = trainingConfiguration.timeScale;
            Application.targetFrameRate = trainingConfiguration.targetFrameRate;
            QualitySettings.vSyncCount = 0;
            Monitor.SetActive(false);
        }
        else
        {
            Screen.SetResolution(inferenceConfiguration.width, inferenceConfiguration.height, false);
            QualitySettings.SetQualityLevel(inferenceConfiguration.qualityLevel, true);
            Time.timeScale = inferenceConfiguration.timeScale;
            Application.targetFrameRate = inferenceConfiguration.targetFrameRate;
        }
    }

    /// Environment specific step logic.
    /**
	 * Implemented in environment-specific child class. 
	 * This method is called at every step. 
	*/
    public virtual void AcademyStep()
    {

    }

    /// Environment specific reset logic.
    /**
	* Implemented in environment-specific child class. 
	* This method is called everytime the Academy resets (when the global done
	* flag is set to true).
	*/
    public virtual void AcademyReset()
    {

    }

    public void MaxStepReached()
    {
        maxStepReached = true;
        Done();
    }
    public void Done()
    {
        done = true;
    }
    public bool IsMaxStepR4eached()
    {
        return maxStepReached;
    }
    public bool IsDone()
    {
        return done;
    }



    // Called after AcademyStep().
    internal void _AcademyStep()
    {

        if (isInference != _isCurrentlyInference)
        {
            ConfigureEngine();
            _isCurrentlyInference = isInference;
        }
        if (communicator != null)
        {
            if (communicator.GetCommand() == ExternalCommand.RESET)
            {
                Dictionary<string, float> NewResetParameters = communicator.GetResetParameters();
                foreach (KeyValuePair<string, float> kv in NewResetParameters)
                {
                    resetParameters[kv.Key] = kv.Value;
                }
                foreach (Agent agent in agents)
                {
                    // TODO : Should all agents be asking for a decision now ?
                    //agent.requestDecision = false;
                    //agent.requestAction = false;
                    //This reset is forced : No flags are set

                }
                _AcademyReset();
                communicator.SetCommand(ExternalCommand.STEP);
                //return;
            }
            if (communicator.GetCommand() == ExternalCommand.QUIT)
            {
                Application.Quit();
                //return;
            }
        }

        if ((stepsSinceReset >= maxSteps) && maxSteps > 0)
        {
            MaxStepReached();
        }
        if (done)
            _AcademyReset();
        foreach (Agent agent in agents)
        {
            if (maxStepReached)
                agent.MaxStepReached();
            if (done)
                agent.Done();
        }
        agentsTerminate.Clear();


        foreach (Agent agent in agents)
        {
            // If an agent is done, then it will also request for a decision and an action
            if (agent.IsDone())
            {
                if (agent.agentParameters.resetOnDone) 
                {
                    if (agent.agentParameters.eventBased)
                    {
                        if (!agentsHasAlreadyReset.Contains(agent))
                        {
                            //If event based, the agent can reset as soon as it is done
                            agent._AgentReset();
                            agentsHasAlreadyReset.Add(agent);
                        }
                    }
                    else if (agent.HasRequestedDecision()){
                        // If not event based, the agent must wait to request a decsion
                        // before reseting to keep multiple agents in sync.
                        agent._AgentReset();
                    }
                }
                else
                {
                    agentsTerminate.Add(agent);
                    agent.RequestDecision();
                }
            }
        }

        foreach (Agent agent in agents)
        {
            if (agent.HasRequestedDecision())
            {
                agent.SendStateToBrain();
                agent._ClearDone();
                agent._ClearMaxStepReached();
                agent.SetReward(0f);
                agent._ClearRequestDecision();
                if(agentsHasAlreadyReset.Contains(agent)){
                    agentsHasAlreadyReset.Remove(agent);
                }
            }

        }
        foreach (Brain brain in brains)
        {
            brain.DecideAction();
        }

        foreach (Agent agent in agentsTerminate)
        {
            agent._ClearRequestAction();
            agent.AgentOnDone();
            UnRegisterAgent(agent);

        }
        foreach (Agent agent in agents)
        {
            agent._AgentStep();
        }
        stepsSinceReset += 1;


    }

    internal void _AcademyReset()
    {
        stepsSinceReset = 0;
        episodeCount++;
        done = false;
        maxStepReached = false;
        AcademyReset();

        //The list might change while iterating. Consider Making a copy.
        foreach (Agent agent in agents)
        {
            agent._AgentReset();
            agent._resetCounters();
        }

    }

    void FixedUpdate()
    {
        _AcademyStep();

    }


    private static void GetBrains(GameObject gameObject, List<Brain> brains)
    {
        var transform = gameObject.transform;

        for (var i = 0; i < transform.childCount; i++)
        {
            var child = transform.GetChild(i);
            var brain = child.GetComponent<Brain>();

            if (brain != null && child.gameObject.activeSelf)
                brains.Add(brain);
        }
    }
}