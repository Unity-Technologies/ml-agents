using System;
using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents.Inference;
using MLAgents.Policies;
using MLAgents.SideChannels;
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
    /// Helper class to step the Academy during FixedUpdate phase.
    /// </summary>
    internal class AcademyFixedUpdateStepper : MonoBehaviour
    {
        void FixedUpdate()
        {
            Academy.Instance.EnvironmentStep();
        }
    }

    /// <summary>
    /// An Academy is where Agent objects go to train their behaviors.
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
        "docs/Learning-Environment-Design.md")]
    public class Academy : IDisposable
    {
        /// <summary>
        /// Communication protocol version.
        /// When connecting to python, this must match UnityEnvironment.API_VERSION.
        /// Currently we require strict equality between the communication protocol
        /// on each side, although we may allow some flexibility in the future.
        /// This should be incremented whenever a change is made to the communication protocol.
        /// </summary>
        const string k_ApiVersion = "0.15.0";

        /// <summary>
        /// Unity package version of com.unity.ml-agents.
        /// This must match the version string in package.json and is checked in a unit test.
        /// </summary>
        internal const string k_PackageVersion = "0.15.0-preview";

        const int k_EditorTrainingPort = 5004;

        const string k_portCommandLineFlag = "--mlagents-port";

        // Lazy initializer pattern, see https://csharpindepth.com/articles/singleton#lazy
        static Lazy<Academy> s_Lazy = new Lazy<Academy>(() => new Academy());

        /// <summary>
        /// True if the Academy is initialized, false otherwise.
        /// </summary>
        public static bool IsInitialized
        {
            get { return s_Lazy.IsValueCreated; }
        }

        /// <summary>
        /// The singleton Academy object.
        /// </summary>
        public static Academy Instance { get { return s_Lazy.Value; } }

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

        // The Academy uses a series of events to communicate with agents
        // to facilitate synchronization. More specifically, it ensure
        // that all the agents performs their steps in a consistent order (i.e. no
        // agent can act based on a decision before another agent has had a chance
        // to request a decision).

        // Signals to all the Agents at each environment step so they can use
        // their Policy to decide on their next action.
        internal event Action DecideAction;

        // Signals to all the listeners that the academy is being destroyed
        internal event Action DestroyAction;

        // Signals the Agent that a new step is about to start.
        // This will mark the Agent as Done if it has reached its maxSteps.
        internal event Action AgentIncrementStep;

        // Signals to all the agents at each environment step along with the
        // Academy's maxStepReached, done and stepCount values. The agents rely
        // on this event to update their own values of max step reached and done
        // in addition to aligning on the step count of the global episode.
        internal event Action<int> AgentSetStatus;

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
        }

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
                if (args[i] == k_portCommandLineFlag)
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
            EnableAutomaticStepping();

            SideChannelUtils.RegisterSideChannel(new EngineConfigurationChannel());
            SideChannelUtils.RegisterSideChannel(new FloatPropertiesChannel());
            SideChannelUtils.RegisterSideChannel(new StatsSideChannel());

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
                        });
                    UnityEngine.Random.InitState(unityRlInitParameters.seed);
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
            AgentSetStatus = i => {};
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
        /// Returns the current episode counter.
        /// </summary>
        /// <returns>
        /// Current episode number.
        /// </returns>
        public int EpisodeCount
        {
            get { return m_EpisodeCount; }
        }

        /// <summary>
        /// Returns the current step counter (within the current episode).
        /// </summary>
        /// <returns>
        /// Current step count.
        /// </returns>
        public int StepCount
        {
            get { return m_StepCount; }
        }

        /// <summary>
        /// Returns the total step counter.
        /// </summary>
        /// <returns>
        /// Total step count.
        /// </returns>
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
        /// Performs a single environment update to the Academy, and Agent
        /// objects within the environment.
        /// </summary>
        public void EnvironmentStep()
        {
            if (!m_HadFirstReset)
            {
                ForcedFullReset();
            }

            AgentSetStatus?.Invoke(m_StepCount);

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
                SideChannelUtils.GetSideChannelMessage();
            }

            using (TimerStack.Instance.Scoped("AgentAct"))
            {
                AgentAct?.Invoke();
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
        /// <param name="brainParameters">The brainParameters used to create the ModelRunner.</param>
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
                modelRunner = new ModelRunner(
                    model, brainParameters, inferenceDevice);
                m_ModelRunners.Add(modelRunner);
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
            SideChannelUtils.UnregisterAllSideChannels();

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
    }
}
