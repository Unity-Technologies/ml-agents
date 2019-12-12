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
        const int k_EditorTrainingPort = 5004;

        /// Temporary storage for global gravity value
        /// Used to restore oringal value when deriving Academy modifies it
        Vector3 m_OriginalGravity;

        /// Temporary storage for global fixedDeltaTime value
        /// Used to restore original value when deriving Academy modifies it
        float m_OriginalFixedDeltaTime;

        /// Temporary storage for global maximumDeltaTime value
        /// Used to restore original value when deriving Academy modifies it
        float m_OriginalMaximumDeltaTime;

        public IFloatProperties FloatProperties;


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
        static int ReadPortFromArgs()
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

            try
            {
                return int.Parse(inputPort);
            }
            catch
            {
                // No arg passed, or malformed port number.
#if UNITY_EDITOR
                // Try connecting on the default editor port
                return k_EditorTrainingPort;
#else
                // This is an executable, so we don't try to connect.
                return -1;
#endif
            }

        }

        /// <summary>
        /// Initializes the environment, configures it and initialized the Academy.
        /// </summary>
        void InitializeEnvironment()
        {
            m_OriginalGravity = Physics.gravity;
            m_OriginalFixedDeltaTime = Time.fixedDeltaTime;
            m_OriginalMaximumDeltaTime = Time.maximumDeltaTime;

            var floatProperties = new FloatPropertiesChannel();
            FloatProperties = floatProperties;
            InitializeAcademy();


            // Try to launch the communicator by using the arguments passed at launch
            var port = ReadPortFromArgs();
            if (port > 0)
            {
                Communicator = new RpcCommunicator(
                    new CommunicatorInitParameters
                    {
                        port = port
                    }
                );
            }

            if (Communicator != null)
            {
                Communicator.RegisterSideChannel(new EngineConfigurationChannel());
                Communicator.RegisterSideChannel(floatProperties);
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
                        });
                    Random.InitState(unityRLInitParameters.seed);
                }
                catch
                {
                    Debug.Log($"" +
                        $"Couldn't connect to trainer on port {port} using API version {k_ApiVersion}. " +
                        "Will perform inference instead."
                    );
                    Communicator = null;
                }

                if (Communicator != null)
                {
                    Communicator.QuitCommandReceived += OnQuitCommandReceived;
                    Communicator.ResetCommandReceived += OnResetCommand;
                }
            }

            // If a communicator is enabled/provided, then we assume we are in
            // training mode. In the absence of a communicator, we assume we are
            // in inference mode.

            DecideAction += () => { };
            DestroyAction += () => { };
            AgentSetStatus += i => { };
            AgentResetIfDone += () => { };
            AgentSendState += () => { };
            AgentAct += () => { };
            AgentForceReset += () => { };

        }

        static void OnQuitCommandReceived()
        {
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#endif
            Application.Quit();
        }

        void OnResetCommand()
        {
            ForcedFullReset();
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
