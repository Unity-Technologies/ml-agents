using System;
using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.SideChannels;
using Unity.Barracuda;

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
 * https://github.com/Unity-Technologies/ml-agents/tree/release_2_verified_docs/docs/
 */

namespace Unity.MLAgents
{
    /// <summary>
    /// Helper class to step the Academy during FixedUpdate phase.
    /// </summary>
    internal class AcademyFixedUpdateStepper : MonoBehaviour
    {
        void FixedUpdate()
        {
            // Check if the stepper belongs to the current Academy and destroy it if it's not.
            // This is to prevent from having leaked stepper from previous runs.
            if (!Academy.IsInitialized || !Academy.Instance.IsStepperOwner(this))
            {
                Destroy(this.gameObject);
            }
            else
            {
                Academy.Instance.EnvironmentStep();
            }
        }
    }

    /// <summary>
    /// The Academy singleton manages agent training and decision making.
    /// </summary>
    /// <remarks>
    /// Access the Academy singleton through the <see cref="Instance"/>
    /// property. The Academy instance is initialized the first time it is accessed (which will
    /// typically be by the first <see cref="Agent"/> initialized in a scene).
    ///
    /// At initialization, the Academy attempts to connect to the Python training process through
    /// the external communicator. If successful, the training process can train <see cref="Agent"/>
    /// instances. When you set an agent's <see cref="BehaviorParameters.BehaviorType"/> setting
    /// to <see cref="BehaviorType.Default"/>, the agent exchanges data with the training process
    /// to make decisions. If no training process is available, agents with the default behavior
    /// fall back to inference or heuristic decisions. (You can also set agents to always use
    /// inference or heuristics.)
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/tree/release_2_verified_docs/" +
        "docs/Learning-Environment-Design.md")]
    public class Academy : IDisposable
    {
        /// <summary>
        /// Communication protocol version.
        /// When connecting to python, this must be compatible with UnityEnvironment.API_VERSION.
        /// We follow semantic versioning on the communication version, so existing
        /// functionality will work as long the major versions match.
        /// This should be changed whenever a change is made to the communication protocol.
        /// </summary>
        const string k_ApiVersion = "1.0.0";

        /// <summary>
        /// Unity package version of com.unity.ml-agents.
        /// This must match the version string in package.json and is checked in a unit test.
        /// </summary>
        internal const string k_PackageVersion = "1.0.8";

        const int k_EditorTrainingPort = 5004;

        const string k_PortCommandLineFlag = "--mlagents-port";

        // Lazy initializer pattern, see https://csharpindepth.com/articles/singleton#lazy
        static Lazy<Academy> s_Lazy = new Lazy<Academy>(() => new Academy());

        /// <summary>
        ///Reports whether the Academy has been initialized yet.
        /// </summary>
        /// <value><c>True</c> if the Academy is initialized, <c>false</c> otherwise.</value>
        public static bool IsInitialized
        {
            get { return s_Lazy.IsValueCreated; }
        }

        /// <summary>
        /// The singleton Academy object.
        /// </summary>
        /// <value>Getting the instance initializes the Academy, if necessary.</value>
        public static Academy Instance { get { return s_Lazy.Value; } }

        // Fields not provided in the Inspector.

        /// <summary>
        /// Reports whether or not the communicator is on.
        /// </summary>
        /// <seealso cref="ICommunicator"/>
        /// <value>
        /// <c>True</c>, if communicator is on, <c>false</c> otherwise.
        /// </value>
        public bool IsCommunicatorOn
        {
            get { return Communicator != null; }
        }

        /// The number of episodes completed by the environment. Incremented
        /// each time the environment is reset.
        int m_EpisodeCount;

        /// The number of steps completed within the current episode. Incremented
        /// each time a step is taken in the environment. Is reset to 0 during
        /// <see cref="EnvironmentReset"/>.
        int m_StepCount;

        /// The number of total number of steps completed during the whole simulation. Incremented
        /// each time a step is taken in the environment.
        int m_TotalStepCount;

        /// Pointer to the communicator currently in use by the Academy.
        internal ICommunicator Communicator;

        bool m_Initialized;
        List<ModelRunner> m_ModelRunners = new List<ModelRunner>();

        // Flag used to keep track of the first time the Academy is reset.
        bool m_HadFirstReset;

        // Detect an Academy step called by user code that is also called by the Academy.
        private RecursionChecker m_StepRecursionChecker = new RecursionChecker("EnvironmentStep");

        // Random seed used for inference.
        int m_InferenceSeed;

        /// <summary>
        /// Set the random seed used for inference. This should be set before any Agents are added
        /// to the scene. The seed is passed to the ModelRunner constructor, and incremented each
        /// time a new ModelRunner is created.
        /// </summary>
        public int InferenceSeed
        {
            set { m_InferenceSeed = value; }
        }

        /// <summary>
        /// Returns the RLCapabilities of the python client that the unity process is connected to.
        /// </summary>
        internal UnityRLCapabilities TrainerCapabilities { get; set; }


        // The Academy uses a series of events to communicate with agents
        // to facilitate synchronization. More specifically, it ensures
        // that all the agents perform their steps in a consistent order (i.e. no
        // agent can act based on a decision before another agent has had a chance
        // to request a decision).

        // Signals to all the Agents at each environment step so they can use
        // their Policy to decide on their next action.
        internal event Action DecideAction;

        // Signals to all the listeners that the academy is being destroyed
        internal event Action DestroyAction;

        // Signals to the Agent that a new step is about to start.
        // This will mark the Agent as Done if it has reached its maxSteps.
        internal event Action AgentIncrementStep;


        /// <summary>
        /// Signals to all of the <see cref="Agent"/>s that their step is about to begin.
        /// This is a good time for an <see cref="Agent"/> to decide if it would like to
        /// call <see cref="Agent.RequestDecision"/> or <see cref="Agent.RequestAction"/>
        /// for this step.  Any other pre-step setup could be done during this even as well.
        /// </summary>
        public event Action<int> AgentPreStep;

        // Signals to all the agents at each environment step so they can send
        // their state to their Policy if they have requested a decision.
        internal event Action AgentSendState;

        // Signals to all the agents at each environment step so they can act if
        // they have requested a decision.
        internal event Action AgentAct;

        // Signals to all the agents each time the Academy force resets.
        internal event Action AgentForceReset;

        /// <summary>
        /// Signals that the Academy has been reset by the training process.
        /// </summary>
        public event Action OnEnvironmentReset;

        AcademyFixedUpdateStepper m_FixedUpdateStepper;
        GameObject m_StepperObject;


        /// <summary>
        /// Private constructor called the first time the Academy is used.
        /// Academy uses this time to initialize internal data
        /// structures, initialize the environment and check for the existence
        /// of a communicator.
        /// </summary>
        Academy()
        {
            Application.quitting += Dispose;

            LazyInitialize();

#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#endif
        }

#if UNITY_EDITOR
        /// <summary>
        /// Clean up the Academy when switching from edit mode to play mode
        /// </summary>
        /// <param name="state">State.</param>
        void HandleOnPlayModeChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingEditMode)
            {
                Dispose();
            }
        }
#endif

        /// <summary>
        /// Initialize the Academy if it hasn't already been initialized.
        /// This method is always safe to call; it will have no effect if the Academy is already
        /// initialized.
        /// </summary>
        internal void LazyInitialize()
        {
            if (!m_Initialized)
            {
                InitializeEnvironment();
                m_Initialized = true;
            }
        }

        /// <summary>
        /// Enable stepping of the Academy during the FixedUpdate phase. This is done by creating
        /// a temporary GameObject with a MonoBehaviour that calls Academy.EnvironmentStep().
        /// </summary>
        void EnableAutomaticStepping()
        {
            if (m_FixedUpdateStepper != null)
            {
                return;
            }

            m_StepperObject = new GameObject("AcademyFixedUpdateStepper");
            // Don't show this object in the hierarchy
            m_StepperObject.hideFlags = HideFlags.HideInHierarchy;
            m_FixedUpdateStepper = m_StepperObject.AddComponent<AcademyFixedUpdateStepper>();
            try
            {
                // This try-catch is because DontDestroyOnLoad cannot be used in Editor Tests
                GameObject.DontDestroyOnLoad(m_StepperObject);
            }
            catch {}
        }

        /// <summary>
        /// Disable stepping of the Academy during the FixedUpdate phase. If this is called, the Academy must be
        /// stepped manually by the user by calling Academy.EnvironmentStep().
        /// </summary>
        void DisableAutomaticStepping()
        {
            if (m_FixedUpdateStepper == null)
            {
                return;
            }

            m_FixedUpdateStepper = null;
            if (Application.isEditor)
            {
                UnityEngine.Object.DestroyImmediate(m_StepperObject);
            }
            else
            {
                UnityEngine.Object.Destroy(m_StepperObject);
            }

            m_StepperObject = null;
        }

        /// <summary>
        /// Determines whether or not the Academy is automatically stepped during the FixedUpdate phase.
        /// </summary>
        /// <value>Set <c>true</c> to enable automatic stepping; <c>false</c> to disable.</value>
        public bool AutomaticSteppingEnabled
        {
            get { return m_FixedUpdateStepper != null; }
            set
            {
                if (value)
                {
                    EnableAutomaticStepping();
                }
                else
                {
                    DisableAutomaticStepping();
                }
            }
        }

        // Used to read Python-provided environment parameters
        static int ReadPortFromArgs()
        {
            var args = Environment.GetCommandLineArgs();
            var inputPort = "";
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == k_PortCommandLineFlag)
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

        EnvironmentParameters m_EnvironmentParameters;
        StatsRecorder m_StatsRecorder;

        /// <summary>
        /// Returns the <see cref="EnvironmentParameters"/> instance. If training
        /// features such as Curriculum Learning or Environment Parameter Randomization are used,
        /// then the values of the parameters generated from the training process can be
        /// retrieved here.
        /// </summary>
        /// <returns></returns>
        public EnvironmentParameters EnvironmentParameters
        {
            get { return m_EnvironmentParameters; }
        }

        /// <summary>
        /// Returns the <see cref="StatsRecorder"/> instance. This instance can be used
        /// to record any statistics from the Unity environment.
        /// </summary>
        /// <returns></returns>
        public StatsRecorder StatsRecorder
        {
            get { return m_StatsRecorder; }
        }

        /// <summary>
        /// Initializes the environment, configures it and initializes the Academy.
        /// </summary>
        void InitializeEnvironment()
        {
            TimerStack.Instance.AddMetadata("communication_protocol_version", k_ApiVersion);
            TimerStack.Instance.AddMetadata("com.unity.ml-agents_version", k_PackageVersion);

            EnableAutomaticStepping();

            SideChannelsManager.RegisterSideChannel(new EngineConfigurationChannel());
            m_EnvironmentParameters = new EnvironmentParameters();
            m_StatsRecorder = new StatsRecorder();

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
                // We try to exchange the first message with Python. If this fails, it means
                // no Python Process is ready to train the environment. In this case, the
                //environment must use Inference.
                try
                {
                    var unityRlInitParameters = Communicator.Initialize(
                        new CommunicatorInitParameters
                        {
                            unityCommunicationVersion = k_ApiVersion,
                            unityPackageVersion = k_PackageVersion,
                            name = "AcademySingleton",
                            CSharpCapabilities = new UnityRLCapabilities()
                        });
                    UnityEngine.Random.InitState(unityRlInitParameters.seed);
                    // We might have inference-only Agents, so set the seed for them too.
                    m_InferenceSeed = unityRlInitParameters.seed;
                    TrainerCapabilities = unityRlInitParameters.TrainerCapabilities;
                    TrainerCapabilities.WarnOnPythonMissingBaseRLCapabilities();
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

            ResetActions();
        }

        void ResetActions()
        {
            DecideAction = () => {};
            DestroyAction = () => {};
            AgentPreStep = i => {};
            AgentSendState = () => {};
            AgentAct = () => {};
            AgentForceReset = () => {};
            OnEnvironmentReset = () => {};
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
        /// The current episode count.
        /// </summary>
        /// <value>
        /// Current episode number.
        /// </value>
        public int EpisodeCount
        {
            get { return m_EpisodeCount; }
        }

        /// <summary>
        /// The current step count (within the current episode).
        /// </summary>
        /// <value>
        /// Current step count.
        /// </value>
        public int StepCount
        {
            get { return m_StepCount; }
        }

        /// <summary>
        /// Returns the total step count.
        /// </summary>
        /// <value>
        /// Total step count.
        /// </value>
        public int TotalStepCount
        {
            get { return m_TotalStepCount; }
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
            m_HadFirstReset = true;
        }

        /// <summary>
        /// Performs a single environment update of the Academy and Agent
        /// objects within the environment.
        /// </summary>
        public void EnvironmentStep()
        {
            using (m_StepRecursionChecker.Start())
            {


                if (!m_HadFirstReset)
                {
                    ForcedFullReset();
                }

                AgentPreStep?.Invoke(m_StepCount);

                m_StepCount += 1;
                m_TotalStepCount += 1;
                AgentIncrementStep?.Invoke();

                using (TimerStack.Instance.Scoped("AgentSendState"))
                {
                    AgentSendState?.Invoke();
                }

                using (TimerStack.Instance.Scoped("DecideAction"))
                {
                    DecideAction?.Invoke();
                }

                // If the communicator is not on, we need to clear the SideChannel sending queue
                if (!IsCommunicatorOn)
                {
                    SideChannelsManager.GetSideChannelMessage();
                }

                using (TimerStack.Instance.Scoped("AgentAct"))
                {
                    AgentAct?.Invoke();
                }
            }
        }

        /// <summary>
        /// Resets the environment, including the Academy.
        /// </summary>
        void EnvironmentReset()
        {
            m_StepCount = 0;
            m_EpisodeCount++;
            OnEnvironmentReset?.Invoke();
        }

        /// <summary>
        /// Creates or retrieves an existing ModelRunner that uses the same
        /// NNModel and the InferenceDevice as provided.
        /// </summary>
        /// <param name="model">The NNModel the ModelRunner must use.</param>
        /// <param name="brainParameters">The BrainParameters used to create the ModelRunner.</param>
        /// <param name="inferenceDevice">
        /// The inference device (CPU or GPU) the ModelRunner will use.
        /// </param>
        /// <returns> The ModelRunner compatible with the input settings.</returns>
        internal ModelRunner GetOrCreateModelRunner(
            NNModel model, BrainParameters brainParameters, InferenceDevice inferenceDevice)
        {
            var modelRunner = m_ModelRunners.Find(x => x.HasModel(model, inferenceDevice));
            if (modelRunner == null)
            {
                modelRunner = new ModelRunner(model, brainParameters, inferenceDevice, m_InferenceSeed);
                m_ModelRunners.Add(modelRunner);
                m_InferenceSeed++;
            }
            return modelRunner;
        }

        /// <summary>
        /// Shut down the Academy.
        /// </summary>
        public void Dispose()
        {
            DisableAutomaticStepping();

            // Signal to listeners that the academy is being destroyed now
            DestroyAction?.Invoke();

            Communicator?.Dispose();
            Communicator = null;

            m_EnvironmentParameters.Dispose();
            m_StatsRecorder.Dispose();
            SideChannelsManager.UnregisterAllSideChannels();  // unregister custom side channels

            if (m_ModelRunners != null)
            {
                foreach (var mr in m_ModelRunners)
                {
                    mr.Dispose();
                }

                m_ModelRunners = null;
            }

            // Clear out the actions so we're not keeping references to any old objects
            ResetActions();

            // TODO - Pass worker ID or some other identifier,
            // so that multiple envs won't overwrite each others stats.
            TimerStack.Instance.SaveJsonTimers();
            m_Initialized = false;

            // Reset the Lazy instance
            s_Lazy = new Lazy<Academy>(() => new Academy());
        }

        /// <summary>
        /// Check if the input AcademyFixedUpdateStepper belongs to this Academy.
        /// </summary>
        internal bool IsStepperOwner(AcademyFixedUpdateStepper stepper)
        {
            return GameObject.ReferenceEquals(stepper.gameObject, Academy.Instance.m_StepperObject);
        }
    }
}
