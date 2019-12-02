using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Serialization;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents.InferenceBrain;
using Barracuda;

/**
 * Welcome to Unity Machine Learning Agents (ML-Agents).
 *
 * The ML-Agents toolkit contains four entities: Academy, Agent, Communicator and
 * Python API. The academy and connected agents live within
 * a learning environment (herein called Environment), while the communicator
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

        [Tooltip("Rendering quality of environment. (Higher is better quality.)")]
        [Range(0, 5)]
        public int qualityLevel;

        [Tooltip("Speed at which environment is run. (Higher is faster.)")]
        [Range(1f, 100f)]
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
    /// An Academy is where Agent objects go to train their behaviors.
    /// Currently, this class is expected to be extended to
    /// implement the desired academy behavior.
    /// </summary>
    /// <remarks>
    /// When an academy is run, it can either be in inference or training mode.
    /// The mode is determined by the presence or absence of a Communicator. In
    /// the presence of a communicator, the academy is run in training mode where
    /// the states and observations of each agent are sent through the
    /// communicator. In the absence of a communicator, the academy is run in
    /// inference mode where the agent behavior is determined by the Policy
    /// attached to it.
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
        "docs/Learning-Environment-Design-Academy.md")]
    public abstract class Academy : MonoBehaviour
    {
        const string k_ApiVersion = "API-12";

        /// Temporary storage for global gravity value
        /// Used to restore oringal value when deriving Academy modifies it
        Vector3 m_OriginalGravity;

        /// Temporary storage for global fixedDeltaTime value
        /// Used to restore original value when deriving Academy modifies it
        float m_OriginalFixedDeltaTime;

        /// Temporary storage for global maximumDeltaTime value
        /// Used to restore original value when deriving Academy modifies it
        float m_OriginalMaximumDeltaTime;

        // Fields provided in the Inspector

        [FormerlySerializedAs("trainingConfiguration")]
        [SerializeField]
        [Tooltip("The engine-level settings which correspond to rendering " +
            "quality and engine speed during Training.")]
        EnvironmentConfiguration m_TrainingConfiguration =
            new EnvironmentConfiguration(80, 80, 1, 100.0f, -1);

        [FormerlySerializedAs("inferenceConfiguration")]
        [SerializeField]
        [Tooltip("The engine-level settings which correspond to rendering " +
            "quality and engine speed during Inference.")]
        EnvironmentConfiguration m_InferenceConfiguration =
            new EnvironmentConfiguration(1280, 720, 5, 1.0f, 60);

        /// <summary>
        /// Contains a mapping from parameter names to float values. They are
        /// used in <see cref="AcademyReset"/> and <see cref="AcademyStep"/>
        /// to modify elements in the environment at reset time.
        /// </summary>
        /// <remarks>
        /// Default reset parameters are specified in the academy Editor, and can
        /// be modified when training by passing a config
        /// dictionary at reset.
        /// </remarks>
        [SerializeField]
        [Tooltip("List of custom parameters that can be changed in the " +
            "environment when it resets.")]
        public ResetParameters resetParameters;
        public CommunicatorObjects.CustomResetParametersProto customResetParameters;

        // Fields not provided in the Inspector.

        /// <summary>
        /// Returns whether or not the communicator is on.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if communicator is on, <c>false</c> otherwise.
        /// </returns>
        public bool IsCommunicatorOn
        {
            get { return Communicator != null; }
        }

        /// If true, the Academy will use inference settings. This field is
        /// initialized in <see cref="Awake"/> depending on the presence
        /// or absence of a communicator. Furthermore, it can be modified during
        /// training via <see cref="SetIsInference"/>.
        bool m_IsInference = true;

        /// The number of episodes completed by the environment. Incremented
        /// each time the environment is reset.
        int m_EpisodeCount;

        /// The number of steps completed within the current episode. Incremented
        /// each time a step is taken in the environment. Is reset to 0 during
        /// <see cref="AcademyReset"/>.
        int m_StepCount;

        /// The number of total number of steps completed during the whole simulation. Incremented
        /// each time a step is taken in the environment.
        int m_TotalStepCount;

        /// Flag that indicates whether the inference/training mode of the
        /// environment was switched by the training process. This impacts the
        /// engine settings at the next environment step.
        bool m_ModeSwitched;

        /// Pointer to the communicator currently in use by the Academy.
        public ICommunicator Communicator;

        bool m_Initialized;
        List<ModelRunner> m_ModelRunners = new List<ModelRunner>();

        // Flag used to keep track of the first time the Academy is reset.
        bool m_FirstAcademyReset;

        // The Academy uses a series of events to communicate with agents
        // to facilitate synchronization. More specifically, it ensure
        // that all the agents performs their steps in a consistent order (i.e. no
        // agent can act based on a decision before another agent has had a chance
        // to request a decision).

        // Signals to all the Agents at each environment step so they can use
        // their Policy to decide on their next action.
        public event System.Action DecideAction;

        // Signals to all the listeners that the academy is being destroyed
        public event System.Action DestroyAction;

        // Signals to all the agents at each environment step along with the
        // Academy's maxStepReached, done and stepCount values. The agents rely
        // on this event to update their own values of max step reached and done
        // in addition to aligning on the step count of the global episode.
        public event System.Action<int> AgentSetStatus;

        // Signals to all the agents at each environment step so they can reset
        // if their flag has been set to done (assuming the agent has requested a
        // decision).
        public event System.Action AgentResetIfDone;

        // Signals to all the agents at each environment step so they can send
        // their state to their Policy if they have requested a decision.
        public event System.Action AgentSendState;

        // Signals to all the agents at each environment step so they can act if
        // they have requested a decision.
        public event System.Action AgentAct;

        // Signals to all the agents each time the Academy force resets.
        public event System.Action AgentForceReset;

        /// <summary>
        /// MonoBehavior function called at the very beginning of environment
        /// creation. Academy uses this time to initialize internal data
        /// structures, initialize the environment and check for the existence
        /// of a communicator.
        /// </summary>
        void Awake()
        {
            LazyInitialization();
        }

        public void LazyInitialization()
        {
            if (!m_Initialized)
            {
                InitializeEnvironment();
                m_Initialized = true;
            }
        }

        // Used to read Python-provided environment parameters
        static int ReadArgs()
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
        void InitializeEnvironment()
        {
            m_OriginalGravity = Physics.gravity;
            m_OriginalFixedDeltaTime = Time.fixedDeltaTime;
            m_OriginalMaximumDeltaTime = Time.maximumDeltaTime;

            InitializeAcademy();

            // Try to launch the communicator by using the arguments passed at launch
            try
            {
                Communicator = new RpcCommunicator(
                    new CommunicatorInitParameters
                    {
                        port = ReadArgs()
                    });
            }
            catch
            {
#if UNITY_EDITOR
                Communicator = new RpcCommunicator(
                    new CommunicatorInitParameters
                    {
                        port = 5004
                    });
#endif
            }

            if (Communicator != null)
            {
                // We try to exchange the first message with Python. If this fails, it means
                // no Python Process is ready to train the environment. In this case, the
                //environment must use Inference.
                try
                {
                    var unityRLInitParameters = Communicator.Initialize(
                        new CommunicatorInitParameters
                        {
                            version = k_ApiVersion,
                            name = gameObject.name,
                            environmentResetParameters = new EnvironmentResetParameters
                            {
                                resetParameters = resetParameters,
                                customResetParameters = customResetParameters
                            }
                        });
                    Random.InitState(unityRLInitParameters.seed);
                }
                catch
                {
                    Communicator = null;
                }

                if (Communicator != null)
                {
                    Communicator.QuitCommandReceived += OnQuitCommandReceived;
                    Communicator.ResetCommandReceived += OnResetCommand;
                    Communicator.RLInputReceived += OnRLInputReceived;
                }
            }

            // If a communicator is enabled/provided, then we assume we are in
            // training mode. In the absence of a communicator, we assume we are
            // in inference mode.

            SetIsInference(!IsCommunicatorOn);

            DecideAction += () => { };
            DestroyAction += () => { };
            AgentSetStatus += i => { };
            AgentResetIfDone += () => { };
            AgentSendState += () => { };
            AgentAct += () => { };
            AgentForceReset += () => { };

            ConfigureEnvironment();
        }

        static void OnQuitCommandReceived()
        {
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#endif
            Application.Quit();
        }

        void OnResetCommand(EnvironmentResetParameters newResetParameters)
        {
            UpdateResetParameters(newResetParameters);
            ForcedFullReset();
        }

        void OnRLInputReceived(UnityRLInputParameters inputParams)
        {
            m_IsInference = !inputParams.isTraining;
        }

        void UpdateResetParameters(EnvironmentResetParameters newResetParameters)
        {
            if (newResetParameters.resetParameters != null)
            {
                foreach (var kv in newResetParameters.resetParameters)
                {
                    resetParameters[kv.Key] = kv.Value;
                }
            }
            customResetParameters = newResetParameters.customResetParameters;
        }

        /// <summary>
        /// Configures the environment settings depending on the training/inference
        /// mode and the corresponding parameters passed in the Editor.
        /// </summary>
        void ConfigureEnvironment()
        {
            if (m_IsInference)
            {
                ConfigureEnvironmentHelper(m_InferenceConfiguration);
                Monitor.SetActive(true);
            }
            else
            {
                ConfigureEnvironmentHelper(m_TrainingConfiguration);
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
        /// Returns the <see cref="m_IsInference"/> flag.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if current mode is inference, <c>false</c> if training.
        /// </returns>
        public bool GetIsInference()
        {
            return m_IsInference;
        }

        /// <summary>
        /// Sets the <see cref="m_IsInference"/> flag to the provided value. If
        /// the new flag differs from the current flag value, this signals that
        /// the environment configuration needs to be updated.
        /// </summary>
        /// <param name="isInference">
        /// Environment mode, if true then inference, otherwise training.
        /// </param>
        public void SetIsInference(bool isInference)
        {
            if (m_IsInference != isInference)
            {
                m_IsInference = isInference;

                // This signals to the academy that at the next environment step
                // the engine configurations need updating to the respective mode
                // (i.e. training vs inference) configuration.
                m_ModeSwitched = true;
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
            return m_EpisodeCount;
        }

        /// <summary>
        /// Returns the current step counter (within the current episode).
        /// </summary>
        /// <returns>
        /// Current step count.
        /// </returns>
        public int GetStepCount()
        {
            return m_StepCount;
        }

        /// <summary>
        /// Returns the total step counter.
        /// </summary>
        /// <returns>
        /// Total step count.
        /// </returns>
        public int GetTotalStepCount()
        {
            return m_TotalStepCount;
        }

        /// <summary>
        /// Forces the full reset. The done flags are not affected. Is either
        /// called the first reset at inference and every external reset
        /// at training.
        /// </summary>
        void ForcedFullReset()
        {
            EnvironmentReset();
            AgentForceReset?.Invoke();
            m_FirstAcademyReset = true;
        }

        /// <summary>
        /// Performs a single environment update to the Academy, and Agent
        /// objects within the environment.
        /// </summary>
        void EnvironmentStep()
        {
            if (m_ModeSwitched)
            {
                ConfigureEnvironment();
                m_ModeSwitched = false;
            }
            if (!m_FirstAcademyReset)
            {
                ForcedFullReset();
            }

            AgentSetStatus?.Invoke(m_StepCount);

            using (TimerStack.Instance.Scoped("AgentResetIfDone"))
            {
                AgentResetIfDone?.Invoke();
            }

            using (TimerStack.Instance.Scoped("AgentSendState"))
            {
                AgentSendState?.Invoke();
            }

            using (TimerStack.Instance.Scoped("DecideAction"))
            {
                DecideAction?.Invoke();
            }

            using (TimerStack.Instance.Scoped("AcademyStep"))
            {
                AcademyStep();
            }

            using (TimerStack.Instance.Scoped("AgentAct"))
            {
                AgentAct?.Invoke();
            }

            m_StepCount += 1;
            m_TotalStepCount += 1;
        }

        /// <summary>
        /// Resets the environment, including the Academy.
        /// </summary>
        void EnvironmentReset()
        {
            m_StepCount = 0;
            m_EpisodeCount++;
            AcademyReset();
        }

        /// <summary>
        /// MonoBehaviour function that dictates each environment step.
        /// </summary>
        void FixedUpdate()
        {
            EnvironmentStep();
        }

        /// <summary>
        /// Creates or retrieves an existing ModelRunner that uses the same
        /// NNModel and the InferenceDevice as provided.
        /// </summary>
        /// <param name="model"> The NNModel the ModelRunner must use </param>
        /// <param name="brainParameters"> The brainParameters used to create
        /// the ModelRunner </param>
        /// <param name="inferenceDevice"> The inference device (CPU or GPU)
        /// the ModelRunner will use </param>
        /// <returns> The ModelRunner compatible with the input settings</returns>
        public ModelRunner GetOrCreateModelRunner(
            NNModel model, BrainParameters brainParameters, InferenceDevice inferenceDevice)
        {
            var modelRunner = m_ModelRunners.Find(x => x.HasModel(model, inferenceDevice));
            if (modelRunner == null)
            {
                modelRunner = new ModelRunner(
                    model, brainParameters, inferenceDevice);
                m_ModelRunners.Add(modelRunner);
            }
            return modelRunner;
        }

        /// <summary>
        /// Cleanup function
        /// </summary>
        protected virtual void OnDestroy()
        {
            Physics.gravity = m_OriginalGravity;
            Time.fixedDeltaTime = m_OriginalFixedDeltaTime;
            Time.maximumDeltaTime = m_OriginalMaximumDeltaTime;

            // Signal to listeners that the academy is being destroyed now
            DestroyAction?.Invoke();

            foreach (var mr in m_ModelRunners)
            {
                mr.Dispose();
            }

            // TODO - Pass worker ID or some other identifier,
            // so that multiple envs won't overwrite each others stats.
            TimerStack.Instance.SaveJsonTimers();
        }
    }
}
