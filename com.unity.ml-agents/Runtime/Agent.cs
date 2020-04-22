using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using MLAgents.Sensors;
using MLAgents.Demonstrations;
using MLAgents.Policies;

namespace MLAgents
{
    /// <summary>
    /// Struct that contains all the information for an Agent, including its
    /// observations, actions and current status.
    /// </summary>
    internal struct AgentInfo
    {
        /// <summary>
        /// Keeps track of the last vector action taken by the Brain.
        /// </summary>
        public float[] storedVectorActions;

        /// <summary>
        /// For discrete control, specifies the actions that the agent cannot take.
        /// An element of the mask array is <c>true</c> if the action is prohibited.
        /// </summary>
        public bool[] discreteActionMasks;

        /// <summary>
        /// The current agent reward.
        /// </summary>
        public float reward;

        /// <summary>
        /// Whether the agent is done or not.
        /// </summary>
        public bool done;

        /// <summary>
        /// Whether the agent has reached its max step count for this episode.
        /// </summary>
        public bool maxStepReached;

        /// <summary>
        /// Episode identifier each agent receives at every reset. It is used
        /// to separate between different agents in the environment.
        /// </summary>
        public int episodeId;
    }

    /// <summary>
    /// Struct that contains the action information sent from the Brain to the
    /// Agent.
    /// </summary>
    internal struct AgentAction
    {
        public float[] vectorActions;
    }

    /// <summary>
    /// An agent is an actor that can observe its environment, decide on the
    /// best course of action using those observations, and execute those actions
    /// within the environment. 
    /// </summary>
    /// <remarks>
    /// Create an agent by extending the Agent class. Add your Agent subclass to
    /// a [GameObject] in the [Unity scene] that serves as the agent's environment.
    /// 
    /// Agents in an environment operate in *steps*. The <see cref="Academy"/> object controls
    /// step progression and each Agent instance tracks its own step count. A step
    /// corresponds to one Unity [FixedUpdate] cycle. At each step, an agent collects
    /// observations, passes them to its decision-making policy, and receives an
    /// action vector in response. (You can make observations at a lower frequency, if desired.)
    /// 
    /// Assign a decision making policy to an agent using a <see cref="BehaviorParameters"/>
    /// component attached to the agent's [GameObject]. The <see cref="BehaviorType"/> setting
    /// determines how decisions are made:
    /// 
    /// * <see cref="BehaviorType.Default"/>: decisions are made by the external process, 
    ///   when connected. Otherwise, decisions are made using inference. If no inference model
    ///   is specified in the BehaviorParameters component, then heuristic decision
    ///   making is used.
    /// * <see cref="BehaviorType.InferenceOnly"/>: decisions are always made using the trained
    ///   model specified in the <see cref="BehaviorParameters"/> component.
    /// * <see cref="BehaviorType.HeuristicOnly"/>: when a decision is needed, the agent's
    ///   <see cref="Heuristic"/> function is called. Your implementation is responsible for
    ///   providing the appropriate action.
    ///
    /// To trigger an agent decision automatically, you can attach a <see cref="DecisionRequester"/>
    /// component to the Agent game object. You can also call the agent's <see cref="RequestDecision"/>
    /// function manually. You only need to call <see cref="RequestDecision"/> when the agent is
    /// in a position to act upon the decision. In many cases, this will be every [FixedUpdate]
    /// callback, but could be less frequent. For example, an agent that hops around its environment
    /// can only take an action when it touches the ground, so several frames might elapse between
    /// one decision and the need for the next.
    ///
    /// Agents make observations using <see cref="ISensor"/> implementations. The ML-Agents
    /// API provides implementations for visual observations (<see cref="CameraSensor"/>)
    /// raycast observations (<see cref="RayPerceptionSensor"/>), and arbitrary 
    /// data observations (<see cref="VectorSensor"/>). You can add the
    /// <see cref="CameraSensorComponent"/> and <see cref="RayPerceptionSensorComponent2D"/> or
    /// <see cref="RayPerceptionSensorComponent3D"/> components to an agent's [GameObject] to use
    /// those sensor types. You can implement the <see cref="CollectObservations(VectorSensor)"/>
    /// function in your Agent subclass to use a vector observation. The Agent class calls this
    /// function before it uses the observation vector to make a decision. (If you only use
    /// visual or raycast observations, you do not need to implement
    /// <see cref="CollectObservations"/>.)
    /// 
    /// Use the <see cref="OnActionReceived"/> function to implement the actions your agent can take,
    /// such as moving to reach a goal or interacting with its environment. Both
    /// <see cref="CollectObservations"/> and <see cref="OnActionReceived"/> are called during the Unity
    /// [FixedUpdate] phase.
    /// 
    /// When you call <see cref="EndEpisode"/> on an agent or the agent reaches its <see cref="maxStep"/> count,
    /// its current episode ends. You can reset the agent -- or remove it from the
    /// environment -- by implementing the <see cref="OnEpisodeBegin"/> function. An agent also
    /// becomes done when the <see cref="Academy"/> resets the environment, which only happens when
    /// the <see cref="Academy"/> receives a reste signal from an external process via the
    /// <see cref="Academy.Communicator"/>.
    /// 
    /// The Agent class extends the Unity [MonoBehaviour] class. You can implement the
    /// standard [MonoBehaviour] functions as needed for your agent. Since an agent's
    /// observations and actions take place during the [FixedUpdate] phase, you should
    /// only use the [MonoBehaviour.Update] function for cosmetic purposes. If you override the [MonoBehaviour]
    /// methods, [OnEnable()] or [OnDisable()], always call the base Agent class implementations.
    /// 
    /// You can implement the <see cref="Heuristic"/> function to specify agent actions using
    /// your own heuristic algorithm. Implementing a heuristic function can be useful
    /// for debugging. For example, you can use keyboard input to select agent actions in
    /// order to manually control an agent's behavior. 
    /// 
    /// Note that you can change the inference model assigned to an agent at any step
    /// by calling <see cref="SetModel"/>.
    /// 
    /// See [Agents] and [Reinforcement Learning in Unity] in the [Unity ML-Agents Toolkit manual] for
    /// more information on creating and training agents.
    /// 
    /// For sample implementations of agent behavior, see the examples available in the
    /// [Unity ML-Agents Toolkit] on Github.
    ///
    /// [MonoBehaviour]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.html
    /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
    /// [Unity scene]: https://docs.unity3d.com/Manual/CreatingScenes.html
    /// [FixedUpdate]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.FixedUpdate.html
    /// [MonoBehaviour.Update]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.Update.html
    /// [OnEnable()]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnEnable.html
    /// [OnDisable()]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnDisable.html]
    /// [OnBeforeSerialize()]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnBeforeSerialize.html
    /// [OnAfterSerialize()]: https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnAfterSerialize.html
    /// [Agents]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md
    /// [Reinforcement Learning in Unity]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design.md
    /// [Unity ML-Agents Toolkit]: https://github.com/Unity-Technologies/ml-agents
    /// [Unity ML-Agents Toolkit manual]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Readme.md
    /// 
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
        "docs/Learning-Environment-Design-Agents.md")]
    [Serializable]
    [RequireComponent(typeof(BehaviorParameters))]
    public class Agent : MonoBehaviour, ISerializationCallbackReceiver
    {
        IPolicy m_Brain;
        BehaviorParameters m_PolicyFactory;

        /// This code is here to make the upgrade path for users using maxStep
        /// easier. We will hook into the Serialization code and make sure that
        /// agentParameters.maxStep and this.maxStep are in sync.
        [Serializable]
        internal struct AgentParameters
        {
            public int maxStep;
        }

        [SerializeField][HideInInspector]
        internal AgentParameters agentParameters;
        [SerializeField][HideInInspector]
        internal bool hasUpgradedFromAgentParameters;

        /// <summary>
        /// The maximum number of steps the agent takes before being done.
        /// </summary>
        /// <value>The maximum steps for an agent to take before it resets; or 0 for
        /// unlimited steps.</value>
        /// <remarks>
        /// The max step value determines the maximum length of an agent's episodes.
        /// Set to a positive integer to limit the episode length to that many steps.
        /// Set to 0 for unlimited episode length. 
        ///
        /// When an episode ends and a new one begins, the Agent object's
        /// <seealso cref="OnEpisodeBegin"/> function is called. You can implement
        /// <see cref="OnEpisodeBegin"/> to reset the agent or remove it from the
        /// environment. An agent's episode can also end if you call its <seealso cref="EndEpisode"/>
        /// method or an external process resets the environment through the <see cref="Academy"/>.
        ///
        /// Consider limiting the number of steps in an episode to avoid wasting time during
        /// training. If you set the max step value to a reasonable estimate of the time it should
        /// take to complete a task, then agents that haven’t succeeded in that time frame will 
        /// reset and start a new training episode rather than continue to fail.
        /// </remarks>
        /// <example>
        /// To use a step limit when training while allowing agents to run without resetting
        /// outside of training, you can set the max step to 0 in <see cref="Initialize"/>
        /// if the <see cref="Academy"/> is not connected to an external process.
        /// <code>
        /// using MLAgents;
        ///
        /// public class MyAgent : Agent
        /// {
        ///     public override void Initialize()
        ///     {
        ///         if (!Academy.Instance.IsCommunicatorOn)
        ///         {
        ///             this.maxStep = 0;
        ///         }
        ///     }
        /// }
        /// </code>
        /// </example>
        [HideInInspector] public int maxStep;

        /// Current Agent information (message sent to Brain).
        AgentInfo m_Info;

        /// Current Agent action (message sent from Brain).
        AgentAction m_Action;

        /// Represents the reward the agent accumulated during the current step.
        /// It is reset to 0 at the beginning of every step.
        /// Should be set to a positive value when the agent performs a "good"
        /// action that we wish to reinforce/reward, and set to a negative value
        /// when the agent performs a "bad" action that we wish to punish/deter.
        /// Additionally, the magnitude of the reward should not exceed 1.0
        float m_Reward;

        /// Keeps track of the cumulative reward in this episode.
        float m_CumulativeReward;

        /// Whether or not the agent requests an action.
        bool m_RequestAction;

        /// Whether or not the agent requests a decision.
        bool m_RequestDecision;

        /// Keeps track of the number of steps taken by the agent in this episode.
        /// Note that this value is different for each agent, and may not overlap
        /// with the step counter in the Academy, since agents reset based on
        /// their own experience.
        int m_StepCount;

        /// Number of times the Agent has completed an episode.
        int m_CompletedEpisodes;

        /// Episode identifier each agent receives. It is used
        /// to separate between different agents in the environment.
        /// This Id will be changed every time the Agent resets.
        int m_EpisodeId;

        /// Whether or not the Agent has been initialized already
        bool m_Initialized;

        /// Keeps track of the actions that are masked at each step.
        DiscreteActionMasker m_ActionMasker;

        /// <summary>
        /// Set of DemonstrationWriters that the Agent will write its step information to.
        /// If you use a DemonstrationRecorder component, this will automatically register its DemonstrationWriter.
        /// You can also add your own DemonstrationWriter by calling
        /// DemonstrationRecorder.AddDemonstrationWriterToAgent()
        /// </summary>
        internal ISet<DemonstrationWriter> DemonstrationWriters = new HashSet<DemonstrationWriter>();

        /// <summary>
        /// List of sensors used to generate observations.
        /// Currently generated from attached SensorComponents, and a legacy VectorSensor
        /// </summary>
        internal List<ISensor> sensors;

        /// <summary>
        /// VectorSensor which is written to by AddVectorObs
        /// </summary>
        internal VectorSensor collectObservationsSensor;

        /// <summary>
        /// Called when the attached <see cref="GameObject"/> becomes enabled and active.
        /// </summary>
        /// <remarks>
        /// This function initializes the Agent instance, if it hasn't been initialized yet.
        /// Always call the base Agent class version of this function if you implement `OnEnable()`
        /// in your own Agent subclasses.
        /// </remarks>
        /// <example>
        /// <code>
        /// protected override void OnEnable()
        /// {
        ///     base.OnEnable();
        ///     // additional OnEnable logic...
        /// }
        /// </code>
        /// </example>
        protected virtual void OnEnable()
        {
            LazyInitialize();
        }

        /// <summary>
        /// Called by Unity immediately before serializing this object.
        /// </summary>
        /// <remarks>
        /// The Agent class uses OnBeforeSerialize() for internal housekeeping. Call the
        /// base class implementation if you need your own custom serialization logic.
        ///
        /// See [OnBeforeSerialize] for more information.
        /// 
        /// [OnBeforeSerialize]: https://docs.unity3d.com/ScriptReference/ISerializationCallbackReceiver.OnAfterDeserialize.html
        /// </remarks>
        /// <example>
        /// <code>
        /// public new void OnBeforeSerialize()
        /// {
        ///     base.OnBeforeSerialize();
        ///     // additional serialization logic...
        /// }
        /// </code>
        /// </example>
        public void OnBeforeSerialize()
        {
            // Manages a serialization upgrade issue from v0.13 to v0.14 where maxStep moved
            // from AgentParameters (since removed) to Agent
            if (maxStep == 0 && maxStep != agentParameters.maxStep && !hasUpgradedFromAgentParameters)
            {
                maxStep = agentParameters.maxStep;
            }
            hasUpgradedFromAgentParameters = true;
        }

        /// <summary>
        /// Called by Unity immediately after deserializing this object.
        /// </summary>
        /// <remarks>
        /// The Agent class uses OnAfterDeserialize() for internal housekeeping. Call the
        /// base class implementation if you need your own custom deserialization logic.
        ///
        /// See [OnAfterDeserialize] for more information.
        /// 
        /// [OnAfterDeserialize]: https://docs.unity3d.com/ScriptReference/ISerializationCallbackReceiver.OnAfterDeserialize.html
        /// </remarks>
        /// <example>
        /// <code>
        /// public new void OnAfterDeserialize()
        /// {
        ///     base.OnAfterDeserialize();
        ///     // additional deserialization logic...
        /// }
        /// </code>
        /// </example>
        public void OnAfterDeserialize()
        {
            // Manages a serialization upgrade issue from v0.13 to v0.14 where maxStep moved
            // from AgentParameters (since removed) to Agent
            if (maxStep == 0 && maxStep != agentParameters.maxStep && !hasUpgradedFromAgentParameters)
            {
                maxStep = agentParameters.maxStep;
            }
            hasUpgradedFromAgentParameters = true;
        }

        /// <summary>
        /// Initializes the agent. Can be safely called multiple times.
        /// </summary>
        /// <remarks>
        /// This function calls your <seealso cref="Initialize"/> implementation, if one exists.
        /// </remarks>
        public void LazyInitialize()
        {
            if (m_Initialized)
            {
                return;
            }
            m_Initialized = true;

            // Grab the "static" properties for the Agent.
            m_EpisodeId = EpisodeIdCounter.GetEpisodeId();
            m_PolicyFactory = GetComponent<BehaviorParameters>();

            m_Info = new AgentInfo();
            m_Action = new AgentAction();
            sensors = new List<ISensor>();

            Academy.Instance.AgentIncrementStep += AgentIncrementStep;
            Academy.Instance.AgentSendState += SendInfo;
            Academy.Instance.DecideAction += DecideAction;
            Academy.Instance.AgentAct += AgentStep;
            Academy.Instance.AgentForceReset += _AgentReset;
            m_Brain = m_PolicyFactory.GeneratePolicy(Heuristic);
            ResetData();
            Initialize();
            InitializeSensors();

            // The first time the Academy resets, all Agents in the scene will be
            // forced to reset through the <see cref="AgentForceReset"/> event.
            // To avoid the Agent resetting twice, the Agents will not begin their
            // episode when initializing until after the Academy had its first reset.
            if (Academy.Instance.TotalStepCount != 0)
            {
                OnEpisodeBegin();
            }
        }

        /// <summary>
        /// The reason that the Agent has been set to "done".
        /// </summary>
        enum DoneReason
        {
            /// <summary>
            /// The <see cref="EndEpisode"/> method was called.
            /// </summary>
            DoneCalled,

            /// <summary>
            /// The max steps for the Agent were reached.
            /// </summary>
            MaxStepReached,

            /// <summary>
            /// The Agent was disabled.
            /// </summary>
            Disabled,
        }

        /// <summary>
        /// Called when the attached <see cref="GameObject"/> becomes disabled and inactive.
        /// </summary>
        /// <remarks>
        /// Always call the base Agent class version of this function if you implement `OnDisable()`
        /// in your own Agent subclasses.
        /// </remarks>
        /// <example>
        /// <code>
        /// protected override void OnDisable()
        /// {
        ///     base.OnDisable();
        ///     // additional OnDisable logic...
        /// }
        /// </code>
        /// </example>
        /// <seealso cref="OnEnable"/>
        protected virtual void OnDisable()
        {
            DemonstrationWriters.Clear();

            // If Academy.Dispose has already been called, we don't need to unregister with it.
            // We don't want to even try, because this will lazily create a new Academy!
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentIncrementStep -= AgentIncrementStep;
                Academy.Instance.AgentSendState -= SendInfo;
                Academy.Instance.DecideAction -= DecideAction;
                Academy.Instance.AgentAct -= AgentStep;
                Academy.Instance.AgentForceReset -= _AgentReset;
            }
            NotifyAgentDone(DoneReason.Disabled);
            m_Brain?.Dispose();
            m_Initialized = false;
        }

        void NotifyAgentDone(DoneReason doneReason)
        {
            if (m_Info.done)
            {
                // The Agent was already marked as Done and should not be notified again
                return;
            }
            m_Info.episodeId = m_EpisodeId;
            m_Info.reward = m_Reward;
            m_Info.done = true;
            m_Info.maxStepReached = doneReason == DoneReason.MaxStepReached;
            if (collectObservationsSensor != null)
            {
                // Make sure the latest observations are being passed to training.
                collectObservationsSensor.Reset();
                CollectObservations(collectObservationsSensor);
            }
            // Request the last decision with no callbacks
            // We request a decision so Python knows the Agent is done immediately
            m_Brain?.RequestDecision(m_Info, sensors);
            ResetSensors();

            // We also have to write any to any DemonstationStores so that they get the "done" flag.
            foreach (var demoWriter in DemonstrationWriters)
            {
                demoWriter.Record(m_Info, sensors);
            }

            if (doneReason != DoneReason.Disabled)
            {
                // We don't want to update the reward stats when the Agent is disabled, because this will make
                // the rewards look lower than they actually are during shutdown.
                m_CompletedEpisodes++;
                UpdateRewardStats();
            }

            m_Reward = 0f;
            m_CumulativeReward = 0f;
            m_RequestAction = false;
            m_RequestDecision = false;
            Array.Clear(m_Info.storedVectorActions, 0, m_Info.storedVectorActions.Length);
        }

        /// <summary>
        /// Updates the Model assigned to this Agent instance.
        /// </summary>
        /// <remarks>
        /// If the agent already has an assigned model, that model is replaced with the 
        /// the provided one. However, if you call this function with arguments that are
        /// identical to the current parameters of the agent, then no changes are made.
        ///
        /// When you set a new model, the Agent instance is marked as done, it is reset, and then
        /// its <see cref="OnEpisodeBegin"/> method is called.
        ///
        /// **Note:** the <paramref name="behaviorName"/> parameter is ignored when not training.
        /// The <paramref name="model"/> and <paramref name="inferenceDevice"/> parameters
        /// are ignored when not using inference.
        /// </remarks>
        /// <param name="behaviorName"> The identifier of the behavior. This
        /// will categorize the agent when training.
        /// </param>
        /// <param name="model"> The model to use for inference.</param>
        /// <param name = "inferenceDevice"> Define the device on which the model
        /// will be run.</param>
        public void SetModel(
            string behaviorName,
            NNModel model,
            InferenceDevice inferenceDevice = InferenceDevice.CPU)
        {
            if (behaviorName == m_PolicyFactory.behaviorName &&
                model == m_PolicyFactory.model &&
                inferenceDevice == m_PolicyFactory.inferenceDevice)
            {
                // If everything is the same, don't make any changes.
                return;
            }
            NotifyAgentDone(DoneReason.Disabled);
            m_PolicyFactory.model = model;
            m_PolicyFactory.inferenceDevice = inferenceDevice;
            m_PolicyFactory.behaviorName = behaviorName;
            ReloadPolicy();
        }

        internal void ReloadPolicy()
        {
            if (!m_Initialized)
            {
                // If we haven't initialized yet, no need to make any changes now; they'll
                // happen in LazyInitialize later.
                return;
            }
            m_Brain?.Dispose();
            m_Brain = m_PolicyFactory.GeneratePolicy(Heuristic);
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
        /// Returns the number of episodes that the Agent has completed (either <see cref="Agent.EndEpisode()"/>
        /// was called, or maxSteps was reached).
        /// </summary>
        /// <returns>
        /// Current episode count.
        /// </returns>
        public int CompletedEpisodes
        {
            get { return m_CompletedEpisodes; }
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <remarks>
        /// This function replaces any rewards given to the agent during the current step.
        /// Use <see cref="AddReward(float)"/> to incrementally change the reward rather than
        /// overriding it.
        ///
        /// Typically, you assign rewards in the Agent subclass's <see cref="OnActionReceived(float[])"/>
        /// implementation after carrying out the received action and evaluating its success.
        /// 
        /// Rewards are used during reinforcement learning; they are ignored during inference.
        ///
        /// See [Agents - Rewards] for general advice on implementing rewards and [Reward Signals]
        /// for information about mixing reward signals from curiosity and Generative Adversarial
        /// Imitation Learning (GAIL) with rewards supplied through this method.
        /// 
        /// [Agents - Rewards]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md#rewards
        /// [Reward Signals]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Reward-Signals.md
        /// </remarks>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
#if DEBUG
            Utilities.DebugCheckNanAndInfinity(reward, nameof(reward), nameof(SetReward));
#endif
            m_CumulativeReward += (reward - m_Reward);
            m_Reward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <remarks>Use a positive reward to reinforce desired behavior. You can use a
        /// negative reward to penalize mistakes. Use <seealso cref="SetReward(float)"/> to
        /// set the reward assigned to the current step with a specific value rather than
        /// increasing or decreasing it.
        ///
        /// Typically, you assign rewards in the Agent subclass's <see cref="OnActionReceived(float[])"/>
        /// implementation after carrying out the received action and evaluating its success.
        /// 
        /// Rewards are used during reinforcement learning; they are ignored during inference.
        ///
        /// See [Agents - Rewards] for general advice on implementing rewards and [Reward Signals]
        /// for information about mixing reward signals from curiosity and Generative Adversarial
        /// Imitation Learning (GAIL) with rewards supplied through this method.
        /// 
        /// [Agents - Rewards]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md#rewards
        /// [Reward Signals]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Reward-Signals.md
        ///</remarks>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
#if DEBUG
            Utilities.DebugCheckNanAndInfinity(increment, nameof(increment), nameof(AddReward));
#endif
            m_Reward += increment;
            m_CumulativeReward += increment;
        }

        /// <summary>
        /// Retrieves the episode reward for the Agent.
        /// </summary>
        /// <returns>The episode reward.</returns>
        public float GetCumulativeReward()
        {
            return m_CumulativeReward;
        }

        void UpdateRewardStats()
        {
            var gaugeName = $"{m_PolicyFactory.behaviorName}.CumulativeReward";
            TimerStack.Instance.SetGauge(gaugeName, GetCumulativeReward());
        }

        /// <summary>
        /// Sets the done flag to true and resets the agent.
        /// </summary>
        /// <seealso cref="OnEpisodeBegin"/>
        public void EndEpisode()
        {
            NotifyAgentDone(DoneReason.DoneCalled);
            _AgentReset();
        }

        /// <summary>
        /// Requests a new decision for this agent.
        /// </summary>
        /// <remarks>
        /// Call `RequestDecision()` whenever an agent needs a decision. You often
        /// want to request a decision every environment step. However, if an agent
        /// cannot use the decision every step, then you can request a decision less
        /// frequently.
        ///
        /// You can add a <seealso cref="DecisionRequester"/> component to the agent's
        /// [GameObject] to drive the agent's decision making. When you use this component,
        /// do not call `RequestDecision()` separately.
        ///
        /// Note that this function calls <seealso cref="RequestAction"/>; you do not need to
        /// call both functions at the same time.
        /// 
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        /// </remarks>
        public void RequestDecision()
        {
            m_RequestDecision = true;
            RequestAction();
        }

        /// <summary>
        /// Requests an action for this agent.
        /// </summary>
        /// <remarks>
        /// Call `RequestAction()` to repeat the previous action returned by the agent's
        /// most recent decision. A new decision is not requested. When you call this function,
        /// the Agent instance invokes <seealso cref="OnActionReceived(float[])"/> with the
        /// existing action vector.
        ///
        /// You can use `RequestAction()` in situations where an agent must take an action
        /// every update, but doesn't need to make a decision as often. For example, an 
        /// agent that moves through its environment might need to apply an action to keep 
        /// moving, but only needs to make a decision to change course or speed occasionally.
        /// 
        /// You can add a <seealso cref="DecisionRequester"/> component to the agent's
        /// [GameObject] to drive the agent's decision making and action frequency. When you
        /// use this component, do not call `RequestAction()` separately.
        ///
        /// Note that <seealso cref="RequestDecision"/> calls `RequestAction()`; you do not need to
        /// call both functions at the same time.
        /// 
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        /// </remarks>
        public void RequestAction()
        {
            m_RequestAction = true;
        }

        /// Helper function that resets all the data structures associated with
        /// the agent. Typically used when the agent is being initialized or reset
        /// at the end of an episode.
        void ResetData()
        {
            var param = m_PolicyFactory.brainParameters;
            m_ActionMasker = new DiscreteActionMasker(param);
            // If we haven't initialized vectorActions, initialize to 0. This should only
            // happen during the creation of the Agent. In subsequent episodes, vectorAction
            // should stay the previous action before the Done(), so that it is properly recorded.
            if (m_Action.vectorActions == null)
            {
                m_Action.vectorActions = new float[param.numActions];
                m_Info.storedVectorActions = new float[param.numActions];
            }
        }

        /// <summary>
        /// Implement `Initialize()` to perform one-time initialization or set up of the
        /// Agent instance.
        /// </summary>
        /// <remarks>
        /// `Initialize()` is called once when the agent is first enabled. If, for example,
        /// the Agent object needs references to other [GameObjects] in the scene, you
        /// can collect and store those references here.
        ///
        /// Note that <seealso cref="OnEpisodeBegin"/> is called at the start of each of
        /// the agent's "episodes". You can use that function for items that need to be reset 
        /// for each episode.
        /// 
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        /// </remarks>
        public virtual void Initialize(){}

        /// <summary>
        /// Implement `Heuristic()` to choose an action for this agent using a custom heuristic.
        /// </summary>
        /// <remarks>
        /// Implement this function to provide custom decision making logic or to support manual
        /// control of an agent using keyboard, mouse, or game controller input.
        ///
        /// Your heuristic implementation can use any decision making logic you specify. Assign decision
        /// values to the float[] array, <paramref cref="actionsOut"/>, passed to your function as a parameter.
        /// Add values to the array at the same indexes as they are used in your 
        /// <seealso cref="OnActionReceived(float[])"/> function, which receives this array and 
        /// implements the corresponding agent behavior. See [Actions] for more information
        /// about agent actions.
        /// 
        /// An agent calls this `Heuristic()` function to make a decision when you set its behavior
        /// type to <see cref="BehaviorType.HeuristicOnly"/>. The agent also calls this function if
        /// you set its behavior type to <see cref="BehaviorType.Default"/> when the
        /// <see cref="Academy"/> is not connected to an external training process and you do not
        /// assign a trained model to the agent.
        /// 
        /// To perform imitation learning, implement manual control of the agent in the `Heuristic()`
        /// function so that you can record the demonstrations required for the imitation learning
        /// algorithms. (Attach a [Demonstration Recorder] component to the agent's [GameObject] to
        /// record the demonstration session to a file.)
        /// 
        /// Even when you don’t plan to use heuristic decisions for an agent or imitation learning,
        /// implementing a simple heuristic function can aid in debugging agent actions and interactions
        /// with its environment.
        ///
        /// [Demonstration Recorder]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Training-Imitation-Learning.md#recording-demonstrations
        /// [Actions]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md#actions
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        /// </remarks>
        /// <example>
        /// The following example illustrates a `Heuristic()` function that provides WASD-style
        /// keyboard control for an agent that can move in two dimensions as well as jump. See
        /// [Input Manager] for more information about the built-in Unity input functions.
        /// You can also use the [Input System package], which provides a more flexible and
        /// configurable input system.
        /// <code>
        ///     public override void Heuristic(float[] actionsOut)
        ///     {
        ///         actionsOut[0] = Input.GetAxis("Horizontal");
        ///         actionsOut[1] = Input.GetKey(KeyCode.Space) ? 1.0f : 0.0f;
        ///         actionsOut[2] = Input.GetAxis("Vertical");
        ///     }
        /// </code>
        /// [Input Manager]: https://docs.unity3d.com/Manual/class-InputManager.html
        /// [Input System package]: https://docs.unity3d.com/Packages/com.unity.inputsystem@1.0/manual/index.html
        /// </example>
        /// <seealso cref="OnActionReceived(float[])"/>
        public virtual void Heuristic(float[] actionsOut)
        {
            Debug.LogWarning("Heuristic method called but not implemented. Returning placeholder actions.");
            Array.Clear(actionsOut, 0, actionsOut.Length);
        }

        /// <summary>
        /// Set up the list of ISensors on the Agent. By default, this will select any
        /// SensorBase's attached to the Agent.
        /// </summary>
        internal void InitializeSensors()
        {
            // Get all attached sensor components
            SensorComponent[] attachedSensorComponents;
            if (m_PolicyFactory.useChildSensors)
            {
                attachedSensorComponents = GetComponentsInChildren<SensorComponent>();
            }
            else
            {
                attachedSensorComponents = GetComponents<SensorComponent>();
            }

            sensors.Capacity += attachedSensorComponents.Length;
            foreach (var component in attachedSensorComponents)
            {
                sensors.Add(component.CreateSensor());
            }

            // Support legacy CollectObservations
            var param = m_PolicyFactory.brainParameters;
            if (param.vectorObservationSize > 0)
            {
                collectObservationsSensor = new VectorSensor(param.vectorObservationSize);
                if (param.numStackedVectorObservations > 1)
                {
                    var stackingSensor = new StackingSensor(
                        collectObservationsSensor, param.numStackedVectorObservations);
                    sensors.Add(stackingSensor);
                }
                else
                {
                    sensors.Add(collectObservationsSensor);
                }
            }

            // Sort the Sensors by name to ensure determinism
            sensors.Sort((x, y) => x.GetName().CompareTo(y.GetName()));

#if DEBUG
            // Make sure the names are actually unique
            for (var i = 0; i < sensors.Count - 1; i++)
            {
                Debug.Assert(
                    !sensors[i].GetName().Equals(sensors[i + 1].GetName()),
                    "Sensor names must be unique.");
            }
#endif
        }

        /// <summary>
        /// Sends the Agent info to the linked Brain.
        /// </summary>
        void SendInfoToBrain()
        {
            if (!m_Initialized)
            {
                throw new UnityAgentsException("Call to SendInfoToBrain when Agent hasn't been initialized." +
                    "Please ensure that you are calling 'base.OnEnable()' if you have overridden OnEnable.");
            }

            if (m_Brain == null)
            {
                return;
            }

            if (m_Info.done)
            {
                Array.Clear(m_Info.storedVectorActions, 0, m_Info.storedVectorActions.Length);
            }
            else
            {
                Array.Copy(m_Action.vectorActions, m_Info.storedVectorActions, m_Action.vectorActions.Length);
            }
            m_ActionMasker.ResetMask();
            UpdateSensors();
            using (TimerStack.Instance.Scoped("CollectObservations"))
            {
                CollectObservations(collectObservationsSensor);
            }
            using (TimerStack.Instance.Scoped("CollectDiscreteActionMasks"))
            {
                if (m_PolicyFactory.brainParameters.vectorActionSpaceType == SpaceType.Discrete)
                {
                    CollectDiscreteActionMasks(m_ActionMasker);
                }
            }
            m_Info.discreteActionMasks = m_ActionMasker.GetMask();

            m_Info.reward = m_Reward;
            m_Info.done = false;
            m_Info.maxStepReached = false;
            m_Info.episodeId = m_EpisodeId;

            m_Brain.RequestDecision(m_Info, sensors);

            // If we have any DemonstrationWriters, write the AgentInfo and sensors to them.
            foreach (var demoWriter in DemonstrationWriters)
            {
                demoWriter.Record(m_Info, sensors);
            }
        }

        void UpdateSensors()
        {
            foreach (var sensor in sensors)
            {
                sensor.Update();
            }
        }

        void ResetSensors()
        {
            foreach (var sensor in sensors)
            {
                sensor.Reset();
            }
        }

        /// <summary>
        /// Implement `CollectObservations()` to collect the vector observations of
        /// the agent for the  step. The agent observation describes the current
        /// environment from the perspective of the agent.
        /// </summary>
        /// <param name="sensor">
        /// The vector observations for the agent.
        /// </param>
        /// <remarks>
        /// An agent's observation is any environment information that helps
        /// the agent achieve its goal. For example, for a fighting agent, its
        /// observation could include distances to friends or enemies, or the
        /// current level of ammunition at its disposal.
        /// 
        /// You can use a combination of vector, visual, and raycast observations for an
        /// agent. If you only use visual or raycast observations, you do not need to
        /// implement a `CollectObservations()` function. 
        /// 
        /// Add vector observations to the <paramref name="sensor"/> parameter passed to
        /// this method by calling the <seealso cref="VectorSensor"/> helper methods:
        ///     - <see cref="VectorSensor.AddObservation(int)"/>
        ///     - <see cref="VectorSensor.AddObservation(float)"/>
        ///     - <see cref="VectorSensor.AddObservation(Vector3)"/>
        ///     - <see cref="VectorSensor.AddObservation(Vector2)"/>
        ///     - <see cref="VectorSensor.AddObservation(Quaternion)"/>
        ///     - <see cref="VectorSensor.AddObservation(bool)"/>
        ///     - <see cref="VectorSensor.AddObservation(IEnumerable{float})"/>
        ///     - <see cref="VectorSensor.AddOneHotObservation(int, int)"/>
        ///
        /// You can use any combination of these helper functions to build the agent's
        /// vector of observations. You must build the vector in the same order
        /// each time `CollectObservations()` is called and the length of the vector
        /// must always be the same. In addition, the length of the observation must
        /// match the <see cref="BrainParameters.vectorObservationSize"/>
        /// attribute of the linked Brain, which is set in the Editor on the
        /// **Behavior Parameters** component attached to the agent's [GameObject].
        ///
        /// For more information about observations, see [Observations and Sensors].
        /// 
        /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
        /// [Observations and Sensors]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md#observations-and-sensors
        /// </remarks>
        public virtual void CollectObservations(VectorSensor sensor)
        {
        }

        /// <summary>
        /// Implement `CollectDiscreteActionMasks()` to collects the masks for discrete
        /// actions. When using discrete actions, the agent will not perform the masked
        /// action.
        /// </summary>
        /// <param name="actionMasker">
        /// The action masker for the agent.
        /// </param>
        /// <remarks>
        /// When using Discrete Control, you can prevent the Agent from using a certain
        /// action by masking it with <see cref="DiscreteActionMasker.SetMask(int, IEnumerable{int})"/>.
        ///
        /// See [Agents - Actions] for more information on masking actions.
        /// 
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <seealso cref="OnActionReceived(float[])"/>
        public virtual void CollectDiscreteActionMasks(DiscreteActionMasker actionMasker)
        {
        }

        /// <summary>
        /// Implement `OnActionReceived()` to specify agent behavior at every step, based
        /// on the provided action.
        /// </summary>
        /// <remarks>
        /// An action is passed to this function in the form of an array vector. Your
        /// implementation must use the array to direct the agent's behavior for the
        /// current step.
        /// 
        /// You decide how many elements you need in the action array to control your
        /// agent and what each element means. For example, if you want to apply a
        /// force to move an agent around the environment, you can arbitrarily pick
        /// three values in the action array to use as the force components. During
        /// training, the agent's  policy learns to set those particular elements of
        /// the array to maximize the training rewards the agent receives. (Of course,
        /// if you implement a <seealso cref="Heuristic"/> function, it must use the same
        /// elements of the action array for the same purpose since there is no learning
        /// involved.)
        /// 
        /// Actions for an agent can be either *Continuous* or *Discrete*. Specify which
        /// type of action space an agent uses, along with the size of the action array,
        /// in the <see cref="BrainParameters"/> of the agent's associated
        /// <see cref="BehaviorParameters"/> component. 
        /// 
        /// When an agent uses the continuous action space, the values in the action
        /// array are floating point numbers. You should clamp the values to the range,
        /// -1..1, to increase numerical stability during training.
        /// 
        /// When an agent uses the discrete action space, the values in the action array
        /// are integers that each represent a specific, discrete action. For example,
        /// you could define a set of discrete actions such as:
        ///
        /// <code>
        /// 0 = Do nothing
        /// 1 = Move one space left
        /// 2 = Move one space right
        /// 3 = Move one space up
        /// 4 = Move one space down
        /// </code>
        /// 
        /// When making a decision, the agent picks one of the five actions and puts the
        /// corresponding integer value in the action vector. For example, if the agent
        /// decided to move left, the action vector parameter would contain an array with
        /// a single element with the value 1.
        ///
        /// You can define multiple sets, or branches, of discrete actions to allow an
        /// agent to perform simultaneous, independent actions. For example, you could
        /// use one branch for movement and another branch for throwing a ball left, right,
        /// up, or down, to allow the agent to do both in the same step. 
        /// 
        /// The action vector of a discrete action space contains one element for each
        /// branch. The value of each element is the integer representing the chosen
        /// action for that branch. The agent always chooses one action for each
        /// branch.
        /// 
        /// When you use the discrete action space, you can prevent the training process
        /// or the neural network model from choosing specific actions in a step by
        /// implementing the <see cref="CollectDiscreteActionMasks(DiscreteActionMasker)"/>
        /// function. For example, if your agent is next to a wall, you could mask out any
        /// actions that would result in the agent trying to move into the wall. 
        /// 
        /// For more information about implementing agent actions see [Agents - Actions]. 
        /// 
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <param name="vectorAction">
        /// An array containing the action vector. The length of the array is specified
        /// by the <see cref="BrainParameters"/> of the agent's associated
        /// <see cref="BehaviorParameters"/> component.
        /// </param>
        public virtual void OnActionReceived(float[] vectorAction){}

        /// <summary>
        /// Implement `OnEpisodeBegin()` to set up an Agent instance at the beginning
        /// of an episode. 
        /// </summary>
        /// <seealso cref="Initialize"/>
        /// <seealso cref="EndEpisode"/>
        public virtual void OnEpisodeBegin() {}

        /// <summary>
        /// Returns the last action that was decided on by the Agent.
        /// </summary>
        /// <returns>
        /// The last action that was decided by the Agent (or null if no decision has been made).
        /// </returns>
        /// <seealso cref="OnActionReceived(float[])"/>
        public float[] GetAction()
        {
            return m_Action.vectorActions;
        }

        /// <summary>
        /// An internal reset method that updates internal data structures in
        /// addition to calling <see cref="OnEpisodeBegin"/>.
        /// </summary>
        void _AgentReset()
        {
            ResetData();
            m_StepCount = 0;
            OnEpisodeBegin();
        }

        /// <summary>
        /// Scales continuous action from [-1, 1] to arbitrary range.
        /// </summary>
        /// <param name="rawAction">The input action value.</param>
        /// <param name="min">The minimum output value.</param>
        /// <param name="max">The maximum output value.</param>
        /// <returns>The <paramref name="rawAction"/> scaled from [-1,1] to
        /// [<paramref name="min"/>, <paramref name="max"/>].</returns>
        protected static float ScaleAction(float rawAction, float min, float max)
        {
            var middle = (min + max) / 2;
            var range = (max - min) / 2;
            return rawAction * range + middle;
        }

        /// <summary>
        /// Signals the agent that it must send its decision to the brain.
        /// </summary>
        void SendInfo()
        {
            // If the Agent is done, it has just reset and thus requires a new decision
            if (m_RequestDecision)
            {
                SendInfoToBrain();
                m_Reward = 0f;
                m_RequestDecision = false;
            }
        }

        void AgentIncrementStep()
        {
            m_StepCount += 1;
        }

        /// Used by the brain to make the agent perform a step.
        void AgentStep()
        {
            if ((m_RequestAction) && (m_Brain != null))
            {
                m_RequestAction = false;
                OnActionReceived(m_Action.vectorActions);
            }

            if ((m_StepCount >= maxStep) && (maxStep > 0))
            {
                NotifyAgentDone(DoneReason.MaxStepReached);
                _AgentReset();
            }
        }

        void DecideAction()
        {
            if (m_Action.vectorActions == null)
            {
                ResetData();
            }
            var action = m_Brain?.DecideAction();

            if (action == null)
            {
                Array.Clear(m_Action.vectorActions, 0, m_Action.vectorActions.Length);
            }
            else
            {
                Array.Copy(action, m_Action.vectorActions, action.Length);
            }
        }
    }
}
