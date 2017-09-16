using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*! \mainpage ML-Agents Index Page
 * Welcome to Unity Machine Learning Agents documentation.
 */


[System.Serializable]
public class ScreenConfiguration
{
    public int width;
    public int height;
    [Range(0, 5)]
    public int qualityLevel;
    [Range(1f, 100f)]
    public float timeScale;
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


/** Create a child class to implement InitializeAcademy(), AcademyStep() 
 * and AcademyReset(). The child class script must be attached to an empty game
 * object in your scene, and there can only be one such object within the scene.
 */
public abstract class Academy : MonoBehaviour
{

    [System.Serializable]
    private struct ResetParameter
    {
        public string key;
        public float value;
    }



    [SerializeField]
    private int maxSteps;
    [SerializeField]
    private int frameToSkip;
    [SerializeField]
    private float waitTime;
    [SerializeField]
    private ScreenConfiguration trainingConfiguration = new ScreenConfiguration(80, 80, 1, 100.0f, 60);
    [SerializeField]
    private ScreenConfiguration inferenceConfiguration = new ScreenConfiguration(1280, 720, 5, 1.0f, 60);
    [SerializeField]
    private ResetParameter[] defaultResetParameters;
    public Dictionary<string, float> resetParameters;
    /**< \brief Contains a mapping from parameter names to float values. */
    /**< You can specify the Default Reset Parameters in the Inspector of the
	 * Academy. You can modify these parameters when training with an External 
	 * brain by passing a config dictionary at reset. Reference resetParameters
	 * in your AcademyReset() or AcademyStep() to modify elements in your 
	 * environment at reset time. */


    [HideInInspector]
    private Brain[] brains = new Brain[0];



    ExternalCommand externalCommand;

    private bool acceptingSteps;
    private int framesSinceAction;
    private bool skippingFrames = true;
    [HideInInspector]
    public bool done;
    /**< \brief The done flag of the Academy. */
    /**< When set to true, the Academy will call AcademyReset() instead of 
	* AcademyStep() at step time.
	* If true, all agents done flags will be set to true.*/
    [HideInInspector]
    public int episodeCount;
    /**< \brief Increments each time the environment is reset. */
    [HideInInspector]
    public int currentStep;
    /**< \brief Increments each time a step is taken in the environment. Is
    * reset to 0 during AcademyReset(). */

    public Communicator communicator;
    /**< \brief Do not modify : pointer to the communicator currently in 
	 * use by the Academy. */
    [HideInInspector]
    public bool isInference;
    /**< \brief Do not modify : If true, the Academy will use inference 
	 * settings. */
    [HideInInspector]
    public bool windowResize;
    /**< \brief Do not modify : Used to determine if the application window 
	 * should be resized at reset. */


    private float timeAtStep;


    void Awake()
    {
        resetParameters = new Dictionary<string, float>();
        foreach (ResetParameter kv in defaultResetParameters)
        {
            resetParameters[kv.key] = kv.value;
        }

        brains = gameObject.GetComponentsInChildren<Brain>();
        InitializeAcademy();

        foreach (Brain brain in brains)
        {
            brain.InitializeBrain();
        }

        if (communicator != null)
        {
            communicator.InitializeCommunicator();
            externalCommand = communicator.GetCommand();
        }
        windowResize = true;
        done = true;
        acceptingSteps = true;
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
        if ((communicator != null) && (!isInference))
        {
            Screen.SetResolution(trainingConfiguration.width, trainingConfiguration.height, false);
            QualitySettings.SetQualityLevel(trainingConfiguration.qualityLevel, true);
            Time.timeScale = trainingConfiguration.timeScale;
            Application.targetFrameRate = trainingConfiguration.targetFrameRate;
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


    // Called after AcademyStep().
    internal void Step()
    {
        // Reset all agents whose flags are set to done.
        foreach (Brain brain in brains)
        {
            // Set all agents to done if academy is done.
            if (done)
            {
                brain.SendDone();
            }
            brain.ResetIfDone();
        }

        SendState();

        foreach (Brain brain in brains)
        {
            brain.ResetDoneAndReward();
        }
    }

    // Called before AcademyReset().
    internal void Reset()
    {
        if (windowResize)
        {
            ConfigureEngine();
            windowResize = false;
        }
        currentStep = 0;
        episodeCount++;
        done = false;

        foreach (Brain brain in brains)
        {
            brain.Reset();
            brain.ResetDoneAndReward();
        }

        AcademyReset();
    }

    // Instructs all brains to collect states from their agents.
    private void SendState()
    {
        foreach (Brain brain in brains)
        {
            brain.SendState();
        }
    }

    // Instructs all brains to process states to produce actions.
    private void DecideAction()
    {
        if (communicator != null)
        {
            communicator.UpdateActions();
        }

        foreach (Brain brain in brains)
        {
            brain.DecideAction();
        }

        framesSinceAction = 0;
    }

    void FixedUpdate()
    {
        if (acceptingSteps)
        {
            RunMdp();
        }
    }

    // Contains logic for taking steps in environment simulation.
    /** Based on presence of communicator, inference mode, and frameSkip, 
     * decides whether the environment should be stepped or reset.
     */
    void RunMdp()
    {
        if (((communicator == null) || isInference) && (timeAtStep + waitTime > Time.time))
        {
            return;
        }

        timeAtStep = Time.time;
        framesSinceAction += 1;

        currentStep += 1;
        if ((currentStep >= maxSteps) && maxSteps > 0)
        {
            done = true;
        }

        if ((framesSinceAction > frameToSkip) || done)
        {
            skippingFrames = false;
            framesSinceAction = 0;
        }
        else
        {
            skippingFrames = true;
        }


        if (skippingFrames == false)
        {

            if (communicator != null)
            {
                if (externalCommand == ExternalCommand.STEP)
                {
                    Step();
                    externalCommand = communicator.GetCommand();
                }
                if (externalCommand == ExternalCommand.RESET)
                {
                    Dictionary<string, float> NewResetParameters = communicator.GetResetParameters();
                    foreach (KeyValuePair<string, float> kv in NewResetParameters)
                    {
                        resetParameters[kv.Key] = kv.Value;
                    }
                    Reset();
                    externalCommand = ExternalCommand.STEP;
                    RunMdp();
                    return;
                }
                if (externalCommand == ExternalCommand.QUIT)
                {
                    Application.Quit();
                    return;
                }
            }
            else
            {
                if (done)
                {
                    Reset();
                    RunMdp();
                    return;
                }
                else
                {
                    Step();
                }
            }

            DecideAction();

        }


        AcademyStep();

        foreach (Brain brain in brains)
        {
            brain.Step();
        }

    }

}