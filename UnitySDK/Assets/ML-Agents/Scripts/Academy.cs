using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;

#endif

/**
 * Welcome to Unity Machine Learning Agents (ML-Agents).
 * 
 * The ML-Agents toolkit contains five entities: Academy, Brain, Agent, Communicator and
 * Python API. The academy, and all its brains and connected agents live within
 * a learning environment (herin called Environment), while the communicator
 * manages the communication between the learning environment and the Python
 * API. For more information on each of these entities, in addition to how to
 * set-up a learning environment and train the behavior of characters in a
 * Unity scene, please browse our documentation pages on GitHub:
 * https://github.com/Unity-Technologies/ml-agents/blob/master/docs/
 */

namespace MLAgents
{
    /// <summary>
    /// Wraps the environment-level parameters that are provided within the
    /// Editor. These parameters can be provided for training and inference
    /// modes separately and represent screen resolution, rendering quality and
    /// frame rate.
    /// </summary>
    [System.Serializable]
    public class EnvironmentConfiguration
    {
        [Tooltip("Width of the environment window in pixels.")]
        public int width;

        [Tooltip("Height of the environment window in pixels.")]
        public int height;

        [Tooltip("Rendering quality of environment. (Higher is better quality.)")] [Range(0, 5)]
        public int qualityLevel;

        [Tooltip("Speed at which environment is run. (Higher is faster.)")] [Range(1f, 100f)]
        public float timeScale;

        [Tooltip("Frames per second (FPS) engine attempts to maintain.")]
        public int targetFrameRate;

        /// Initializes a new instance of the 
        /// <see cref="EnvironmentConfiguration"/> class.
        /// <param name="width">Width of environment window (pixels).</param>
        /// <param name="height">Height of environment window (pixels).</param>
        /// <param name="qualityLevel">
        /// Rendering quality of environment. Ranges from 0 to 5, with higher.
        /// </param>
        /// <param name="timeScale">
        /// Speed at which environment is run. Ranges from 1 to 100, with higher
        /// values representing faster speed.
        /// </param>
        /// <param name="targetFrameRate">
        /// Target frame rate (per second) that the engine tries to maintain.
        /// </param>
        public EnvironmentConfiguration(
            int width, int height, int qualityLevel,
            float timeScale, int targetFrameRate)
        {
            this.width = width;
            this.height = height;
            this.qualityLevel = qualityLevel;
            this.timeScale = timeScale;
            this.targetFrameRate = targetFrameRate;
        }
    }

    /// <summary>
    /// An Academy is where Agent objects go to train their behaviors. More 
    /// specifically, an academy is a collection of Brain objects and each agent
    /// in a scene is attached to one brain (a single brain may be attached to 
    /// multiple agents). Currently, this class is expected to be extended to
    /// implement the desired academy behavior.
    /// </summary>
    /// <remarks>
    /// When an academy is run, it can either be in inference or training mode.
    /// The mode is determined by the presence or absence of a Communicator. In
    /// the presence of a communicator, the academy is run in training mode where
    /// the states and observations of each agent are sent through the
    /// communicator. In the absence of a communciator, the academy is run in
    /// inference mode where the agent behavior is determined by the brain
    /// attached to it (which may be internal, heuristic or player).
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
             "docs/Learning-Environment-Design-Academy.md")]
    public abstract class Academy : MonoBehaviour
    {
        private const string kApiVersion = "API-5";

        // Fields provided in the Inspector

        [SerializeField]
        [Tooltip("Total number of steps per global episode.\nNon-positive " +
                 "values correspond to episodes without a maximum number of \n" +
                 "steps. Once the step counter reaches this maximum value, the " +
                 "environment will reset.")]
        int maxSteps;

        [SerializeField]
        [Tooltip("The engine-level settings which correspond to rendering " +
                 "quality and engine speed during Training.")]
        EnvironmentConfiguration trainingConfiguration =
            new EnvironmentConfiguration(80, 80, 1, 100.0f, -1);

        [SerializeField]
        [Tooltip("The engine-level settings which correspond to rendering " +
                 "quality and engine speed during Inference.")]
        EnvironmentConfiguration inferenceConfiguration =
            new EnvironmentConfiguration(1280, 720, 5, 1.0f, 60);

        /// <summary>
        /// Contains a mapping from parameter names to float values. They are
        /// used in <see cref="AcademyReset"/> and <see cref="AcademyStep"/>
        /// to modify elements in the environment at reset time.
        /// <summary/>
        /// <remarks>
        /// Default reset parameters are specified in the academy Editor, and can
        /// be modified when training with an external Brain by passinga config 
        /// dictionary at reset. 
        /// </remarks>
        [SerializeField]
        [Tooltip("List of custom parameters that can be changed in the " +
                 "environment when it resets.")]
        public ResetParameters resetParameters;

        // Fields not provided in the Inspector.

        /// Boolean flag indicating whether a communicator is accessible by the
        /// environment. This also specifies whether the environment is in
        /// Training or Inference mode.
        bool isCommunicatorOn;

        /// Keeps track of the id of the last communicator message received.
        /// Remains 0 if there are no communicators. Is used to ensure that
        /// the same message is not used multiple times.
        private ulong lastCommunicatorMessageNumber;

        /// If true, the Academy will use inference settings. This field is 
        /// initialized in <see cref="Awake"/> depending on the presence
        /// or absence of a communicator. Furthermore, it can be modified by an
        /// external Brain during reset via <see cref="SetIsInference"/>.
        bool isInference = true;

        /// The done flag of the academy. When set to true, the academy will
        /// call <see cref="AcademyReset"/> instead of <see cref="AcademyStep"/>
        /// at step time. If true, all agents done flags will be set to true.
        bool done;

        /// Whether the academy has reached the maximum number of steps for the
        /// current episode.
        bool maxStepReached;

        /// The number of episodes completed by the environment. Incremented 
        /// each time the environment is reset.
        int episodeCount;

        /// The number of steps completed within the current episide. Incremented
        /// each time a step is taken in the environment. Is reset to 0 during 
        /// <see cref="AcademyReset"/>.
        int stepCount;

        /// Flag that indicates whether the inference/training mode of the
        /// environment was switched by the external Brain. This impacts the
        /// engine settings at the next environment step.
        bool modeSwitched;

        /// Pointer to the batcher currently in use by the Academy.
        MLAgents.Batcher brainBatcher;

        /// Used to write error messages.
        StreamWriter logWriter;

        /// The path to where the log should be written.
        string logPath;


        // Flag used to keep track of the first time the Academy is reset.
        bool firstAcademyReset;

        // The Academy uses a series of events to communicate with agents and
        // brains to facilitate synchronization. More specifically, it ensure
        // that all the agents performs their steps in a consistent order (i.e. no
        // agent can act based on a decision before another agent has had a chance 
        // to request a decision).

        // Signals to all the Brains at each environment step so they can decide 
        // actions for their agents.
        public event System.Action BrainDecideAction;

        // Signals to all the agents at each environment step along with the 
        // Academy's maxStepReached, done and stepCount values. The agents rely
        // on this event to update their own values of max step reached and done
        // in addition to aligning on the step count of the global episode.
        public event System.Action<bool, bool, int> AgentSetStatus;

        // Signals to all the agents at each environment step so they can reset
        // if their flag has been set to done (assuming the agent has requested a 
        // decision).
        public event System.Action AgentResetIfDone;

        // Signals to all the agents at each environment step so they can send
        // their state to their Brain if they have requested a decision.
        public event System.Action AgentSendState;

        // Signals to all the agents at each environment step so they can act if
        // they have requested a decision.
        public event System.Action AgentAct;

        // Sigals to all the agents each time the Academy force resets.
        public event System.Action AgentForceReset;

        /// <summary>
        /// Monobehavior function called at the very beginning of environment
        /// creation. Academy uses this time to initialize internal data
        /// structures, initialize the environment and check for the existence
        /// of a communicator.
        /// </summary>
        void Awake()
        {
            InitializeEnvironment();
        }

        // Used to read Python-provided environment parameters
        private int ReadArgs()
        {
            var args = System.Environment.GetCommandLineArgs();
            var inputPort = "";
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == "--port")
                {
                    inputPort = args[i + 1];
                }
            }

            return int.Parse(inputPort);
        }

        /// <summary>
        /// Initializes the environment, configures it and initialized the Academy.
        /// </summary>
        private void InitializeEnvironment()
        {
            // Retrieve Brain and initialize Academy
            var brains = GetBrains(gameObject);
            InitializeAcademy();
            MLAgents.Communicator communicator = null;

            // Try to launch the communicator by usig the arguments passed at launch
            try
            {
                communicator = new MLAgents.RPCCommunicator(
                    new MLAgents.CommunicatorParameters
                    {
                        port = ReadArgs()
                    });
            }
            // If it fails, we check if there are any external brains in the scene
            // If there are : Launch the communicator on the default port
            // If there arn't, there is no need for a communicator and it is set
            // to null
            catch
            {
                communicator = null;
                var externalBrain = brains.FirstOrDefault(b => b.brainType == BrainType.External);
                if (externalBrain != null)
                {
                    communicator = new MLAgents.RPCCommunicator(
                        new MLAgents.CommunicatorParameters
                        {
                            port = 5005
                        });
                }
            }

            brainBatcher = new MLAgents.Batcher(communicator);

            // Initialize Brains and communicator (if present)
            foreach (var brain in brains)
            {
                brain.InitializeBrain(this, brainBatcher);
            }

            if (communicator != null)
            {
                isCommunicatorOn = true;

                var academyParameters =
                    new MLAgents.CommunicatorObjects.UnityRLInitializationOutput();
                academyParameters.Name = gameObject.name;
                academyParameters.Version = kApiVersion;
                foreach (var brain in brains)
                {
                    var bp = brain.brainParameters;
                    academyParameters.BrainParameters.Add(
                        MLAgents.Batcher.BrainParametersConvertor(
                            bp,
                            brain.gameObject.name,
                            (MLAgents.CommunicatorObjects.BrainTypeProto)
                            brain.brainType));
                }
                

                academyParameters.EnvironmentParameters =
                    new MLAgents.CommunicatorObjects.EnvironmentParametersProto();
                foreach (var key in resetParameters.Keys)
                {
                    academyParameters.EnvironmentParameters.FloatParameters.Add(
                        key, resetParameters[key]
                    );
                }

                var pythonParameters = brainBatcher.SendAcademyParameters(academyParameters);
                Random.InitState(pythonParameters.Seed);
                Application.logMessageReceived += HandleLog;
                logPath = Path.GetFullPath(".") + "/UnitySDK.log";
                logWriter = new StreamWriter(logPath, false);
                logWriter.WriteLine(System.DateTime.Now.ToString());
                logWriter.WriteLine(" ");
                logWriter.Close();
            }

            // If a communicator is enabled/provided, then we assume we are in
            // training mode. In the absence of a communicator, we assume we are
            // in inference mode.
            isInference = !isCommunicatorOn;

            BrainDecideAction += () => { };
            AgentSetStatus += (m, d, i) => { };
            AgentResetIfDone += () => { };
            AgentSendState += () => { };
            AgentAct += () => { };
            AgentForceReset += () => { };

            // Configure the environment using the configurations provided by
            // the developer in the Editor.
            SetIsInference(!brainBatcher.GetIsTraining());
            ConfigureEnvironment();
        }

        private void UpdateResetParameters()
        {
            var newResetParameters = brainBatcher.GetEnvironmentParameters();
            if (newResetParameters != null)
            {
                foreach (var kv in newResetParameters.FloatParameters)
                {
                    resetParameters[kv.Key] = kv.Value;
                }
            }
        }

        void HandleLog(string logString, string stackTrace, LogType type)
        {
            logWriter = new StreamWriter(logPath, true);
            logWriter.WriteLine(type.ToString());
            logWriter.WriteLine(logString);
            logWriter.WriteLine(stackTrace);
            logWriter.Close();
        }

        /// <summary>
        /// Configures the environment settings depending on the training/inference
        /// mode and the corresponding parameters passed in the Editor.
        /// </summary>
        void ConfigureEnvironment()
        {
            if (isInference)
            {
                ConfigureEnvironmentHelper(inferenceConfiguration);
                Monitor.SetActive(true);
            }
            else
            {
                ConfigureEnvironmentHelper(trainingConfiguration);
                Monitor.SetActive(false);
            }
        }

        /// <summary>
        /// Helper method for initializing the environment based on the provided
        /// configuration.
        /// </summary>
        /// <param name="config">
        /// Environment configuration (specified in the Editor).
        /// </param>
        static void ConfigureEnvironmentHelper(EnvironmentConfiguration config)
        {
            Screen.SetResolution(config.width, config.height, false);
            QualitySettings.SetQualityLevel(config.qualityLevel, true);
            Time.timeScale = config.timeScale;
            Time.captureFramerate = 60;
            Application.targetFrameRate = config.targetFrameRate;
        }

        /// <summary>
        /// Initializes the academy and environment. Called during the waking-up
        /// phase of the environment before any of the scene objects/agents have
        /// been initialized.
        /// </summary>
        public virtual void InitializeAcademy()
        {
        }

        /// <summary>
        /// Specifies the academy behavior at every step of the environment.
        /// </summary>
        public virtual void AcademyStep()
        {
        }

        /// <summary>
        /// Specifies the academy behavior when being reset (i.e. at the completion
        /// of a global episode).
        /// </summary>
        public virtual void AcademyReset()
        {
        }

        /// <summary>
        /// Returns the <see cref="isInference"/> flag.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if current mode is inference, <c>false</c> if training.
        /// </returns>
        public bool GetIsInference()
        {
            return isInference;
        }

        /// <summary>
        /// Sets the <see cref="isInference"/> flag to the provided value. If
        /// the new flag differs from the current flag value, this signals that
        /// the environment configuration needs to be updated.
        /// </summary>
        /// <param name="isInference">
        /// Environment mode, if true then inference, otherwise training.
        /// </param>
        public void SetIsInference(bool isInference)
        {
            if (this.isInference != isInference)
            {
                this.isInference = isInference;

                // This signals to the academy that at the next environment step
                // the engine configurations need updating to the respective mode
                // (i.e. training vs inference) configuraiton.
                modeSwitched = true;
            }
        }

        /// <summary>
        /// Returns the current episode counter.
        /// </summary>
        /// <returns>
        /// Current episode number.
        /// </returns>
        public int GetEpisodeCount()
        {
            return episodeCount;
        }

        /// <summary>
        /// Returns the current step counter (within the current epside).
        /// </summary>
        /// <returns>
        /// Current episode number.
        /// </returns>
        public int GetStepCount()
        {
            return stepCount;
        }

        /// <summary>
        /// Sets the done flag to true.
        /// </summary>
        public void Done()
        {
            done = true;
        }

        /// <summary>
        /// Returns whether or not the academy is done.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if academy is done, <c>false</c> otherwise.
        /// </returns>
        public bool IsDone()
        {
            return done;
        }

        /// <summary>
        /// Returns whether or not the communicator is on.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if communicator is on, <c>false</c> otherwise.
        /// </returns>
        public bool IsCommunicatorOn()
        {
            return isCommunicatorOn;
        }

        /// <summary>
        /// Forces the full reset. The done flags are not affected. Is either 
        /// called the first reset at inference and every external reset
        /// at training.
        /// </summary>
        void ForcedFullReset()
        {
            EnvironmentReset();
            AgentForceReset();
            firstAcademyReset = true;
        }

        /// <summary>
        /// Performs a single environment update to the Academy, Brain and Agent
        /// objects within the environment.
        /// </summary>
        void EnvironmentStep()
        {
            if (modeSwitched)
            {
                ConfigureEnvironment();
                modeSwitched = false;
            }

            if ((isCommunicatorOn) &&
                (lastCommunicatorMessageNumber != brainBatcher.GetNumberMessageReceived()))
            {
                lastCommunicatorMessageNumber = brainBatcher.GetNumberMessageReceived();
                if (brainBatcher.GetCommand() ==
                    MLAgents.CommunicatorObjects.CommandProto.Reset)
                {
                    UpdateResetParameters();

                    SetIsInference(!brainBatcher.GetIsTraining());

                    ForcedFullReset();
                }

                if (brainBatcher.GetCommand() ==
                    MLAgents.CommunicatorObjects.CommandProto.Quit)
                {
#if UNITY_EDITOR
                    EditorApplication.isPlaying = false;
#endif
                    Application.Quit();
                    return;
                }
            }
            else if (!firstAcademyReset)
            {
                UpdateResetParameters();
                ForcedFullReset();
            }

            if ((stepCount >= maxSteps) && maxSteps > 0)
            {
                maxStepReached = true;
                Done();
            }

            AgentSetStatus(maxStepReached, done, stepCount);

            brainBatcher.RegisterAcademyDoneFlag(done);

            if (done)
            {
                EnvironmentReset();
            }

            AgentResetIfDone();

            AgentSendState();

            BrainDecideAction();

            AcademyStep();

            AgentAct();

            stepCount += 1;
        }

        /// <summary>
        /// Resets the environment, including the Academy.
        /// </summary>
        void EnvironmentReset()
        {
            stepCount = 0;
            episodeCount++;
            done = false;
            maxStepReached = false;
            AcademyReset();
        }

        /// <summary>
        /// Monobehavior function that dictates each environment step.
        /// </summary>
        void FixedUpdate()
        {
            EnvironmentStep();
        }

        /// <summary>
        /// Helper method that retrieves the Brain objects that are currently
        /// specified as children of the Academy within the Editor.
        /// </summary>
        /// <param name="academy">Academy.</param>
        /// <returns>
        /// List of brains currently attached to academy.
        /// </returns>
        static List<Brain> GetBrains(GameObject academy)
        {
            List<Brain> brains = new List<Brain>();
            var transform = academy.transform;

            for (var i = 0; i < transform.childCount; i++)
            {
                var child = transform.GetChild(i);
                var brain = child.GetComponent<Brain>();

                if (brain != null && child.gameObject.activeSelf)
                {
                    brains.Add(brain);
                }
            }

            return brains;
        }
    }
}