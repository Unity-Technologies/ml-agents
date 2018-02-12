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

    public event System.Action BrainDecideAction;
    public event System.Action<bool, bool, int> AgentSetStatus;
    public event System.Action AgentResetIfDone;
    public event System.Action AgentSendState;
    public event System.Action AgentAct;
    public event System.Action AgentForceReset;

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

    /**< \brief Do not modify : pointer to the communicator currently in 
     * use by the Academy. */
    public Communicator communicator;



    void Awake()
    {
        _InitializeAcademy();
    }
    void _InitializeAcademy()
    {

        List<Brain> brains = GetBrains(gameObject);
        InitializeAcademy();

        communicator = new ExternalCommunicator(this);
        if (!communicator.CommunicatorHandShake())
        {
            communicator = null;
        }

        foreach (Brain brain in brains)
        {
            brain.InitializeBrain(this, communicator);
        }
        if (communicator != null)
        {
            communicator.InitializeCommunicator();
            communicator.UpdateCommand();
        }

        isInference = (communicator == null);
        _isCurrentlyInference = !isInference;
        done = true;

        BrainDecideAction += () => { };
        AgentSetStatus += (m, d, i) => { };
        AgentResetIfDone += () => { };
        AgentSendState += () => { };
        AgentAct += () => { };
        AgentForceReset += () => { };
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

   
    public void Done()
    {
        done = true;
    }
    public bool IsDone()
    {
        return done;
    }


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
                _AcademyReset();
                communicator.SetCommand(ExternalCommand.STEP);
            }
            if (communicator.GetCommand() == ExternalCommand.QUIT)
            {
                Application.Quit();
            }
        }

        if ((stepsSinceReset >= maxSteps) && maxSteps > 0)
        {
            maxStepReached = true;
            Done();
        }
        if (done)
            _AcademyReset();

        AgentSetStatus(maxStepReached, done, stepsSinceReset);

        AgentResetIfDone();

        AgentSendState();

        BrainDecideAction();

        AgentAct();

        stepsSinceReset += 1;


    }

    internal void _AcademyReset()
    {
        stepsSinceReset = 0;
        episodeCount++;
        done = false;
        maxStepReached = false;
        AcademyReset();

        AgentForceReset();

    }

    void FixedUpdate()
    {
        _AcademyStep();

    }


    private static List<Brain> GetBrains(GameObject gameObject)
    {
        List<Brain> brains = new List<Brain>();
        var transform = gameObject.transform;

        for (var i = 0; i < transform.childCount; i++)
        {
            var child = transform.GetChild(i);
            var brain = child.GetComponent<Brain>();

            if (brain != null && child.gameObject.activeSelf)
                brains.Add(brain);
        }
        return brains;
    }
}