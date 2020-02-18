using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// Struct that contains all the information for an Agent, including its
    /// observations, actions and current status, that is sent to the Brain.
    /// </summary>
    public struct AgentInfo
    {
        /// <summary>
        /// Keeps track of the last vector action taken by the Brain.
        /// </summary>
        public float[] storedVectorActions;

        /// <summary>
        /// For discrete control, specifies the actions that the agent cannot take. Is true if
        /// the action is masked.
        /// </summary>
        public bool[] actionMasks;

        /// <summary>
        /// Current agent reward.
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
    /// Agent Monobehavior class that is attached to a Unity GameObject, making it
    /// an Agent. An agent produces observations and takes actions in the
    /// environment. Observations are determined by the cameras attached
    /// to the agent in addition to the vector observations implemented by the
    /// user in <see cref="CollectObservations"/>. On the other hand, actions
    /// are determined by decisions produced by a Policy. Currently, this
    /// class is expected to be extended to implement the desired agent behavior.
    /// </summary>
    /// <remarks>
    /// Simply speaking, an agent roams through an environment and at each step
    /// of the environment extracts its current observation, sends them to its
    /// policy and in return receives an action. In practice,
    /// however, an agent need not send its observation at every step since very
    /// little may have changed between successive steps.
    ///
    /// At any step, an agent may be considered <see cref="m_Done"/>.
    /// This could occur due to a variety of reasons:
    ///     - The agent reached an end state within its environment.
    ///     - The agent reached the maximum # of steps (i.e. timed out).
    ///     - The academy reached the maximum # of steps (forced agent to be done).
    ///
    /// Here, an agent reaches an end state if it completes its task successfully
    /// or somehow fails along the way. In the case where an agent is done before
    /// the academy, it either resets and restarts, or just lingers until the
    /// academy is done.
    ///
    /// An important note regarding steps and episodes is due. Here, an agent step
    /// corresponds to an academy step, which also corresponds to Unity
    /// environment step (i.e. each FixedUpdate call). This is not the case for
    /// episodes. The academy controls the global episode count and each agent
    /// controls its own local episode count and can reset and start a new local
    /// episode independently (based on its own experience). Thus an academy
    /// (global) episode can be viewed as the upper-bound on an agents episode
    /// length and that within a single global episode, an agent may have completed
    /// multiple local episodes. Consequently, if an agent max step is
    /// set to a value larger than the academy max steps value, then the academy
    /// value takes precedence (since the agent max step will never be reached).
    ///
    /// Lastly, note that at any step the policy to the agent is allowed to
    /// change model with <see cref="GiveModel"/>.
    ///
    /// Implementation-wise, it is required that this class is extended and the
    /// virtual methods overridden. For sample implementations of agent behavior,
    /// see the Examples/ directory within this Unity project.
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
        "docs/Learning-Environment-Design-Agents.md")]
    [Serializable]
    [RequireComponent(typeof(BehaviorParameters))]
    public abstract class Agent : MonoBehaviour, ISerializationCallbackReceiver
    {
        IPolicy m_Brain;
        BehaviorParameters m_PolicyFactory;

        /// This code is here to make the upgrade path for users using maxStep
        /// easier.  We will hook into the Serialization code and make sure that
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
        /// <remarks>
        /// If set to 0, the agent can only be set to done programmatically (or
        /// when the Academy is done).
        /// If set to any positive integer, the agent will be set to done after
        /// that many steps. Note that setting the max step to a value greater
        /// than the academy max step value renders it useless.
        /// </remarks>
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

        /// Episode identifier each agent receives. It is used
        /// to separate between different agents in the environment.
        /// This Id will be changed every time the Agent resets.
        int m_EpisodeId;

        /// Whether or not the Agent has been initialized already
        bool m_Initialized;

        /// Keeps track of the actions that are masked at each step.
        ActionMasker m_ActionMasker;

        /// <summary>
        /// Demonstration recorder.
        /// </summary>
        DemonstrationRecorder m_Recorder;

        /// <summary>
        /// List of sensors used to generate observations.
        /// Currently generated from attached SensorComponents, and a legacy VectorSensor
        /// </summary>
        internal List<ISensor> sensors;

        /// <summary>
        /// VectorSensor which is written to by AddVectorObs
        /// </summary>
        internal VectorSensor collectObservationsSensor;

        /// MonoBehaviour function that is called when the attached GameObject
        /// becomes enabled or active.
        void OnEnable()
        {
            LazyInitialize();
        }

        public void OnBeforeSerialize()
        {
            if (maxStep == 0 && maxStep != agentParameters.maxStep && !hasUpgradedFromAgentParameters)
            {
                maxStep = agentParameters.maxStep;
            }
            hasUpgradedFromAgentParameters = true;
        }

        public void OnAfterDeserialize()
        {
            if (maxStep == 0 && maxStep != agentParameters.maxStep && !hasUpgradedFromAgentParameters)
            {
                maxStep = agentParameters.maxStep;
            }
            hasUpgradedFromAgentParameters = true;
        }

        /// Helper method for the <see cref="OnEnable"/> event, created to
        /// facilitate testing.
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
            m_Recorder = GetComponent<DemonstrationRecorder>();


            m_Info = new AgentInfo();
            m_Action = new AgentAction();
            sensors = new List<ISensor>();

            Academy.Instance.AgentSendState += SendInfo;
            Academy.Instance.DecideAction += DecideAction;
            Academy.Instance.AgentAct += AgentStep;
            Academy.Instance.AgentForceReset += _AgentReset;
            m_Brain = m_PolicyFactory.GeneratePolicy(Heuristic);
            ResetData();
            InitializeAgent();
            InitializeSensors();
        }

        /// Monobehavior function that is called when the attached GameObject
        /// becomes disabled or inactive.
        void OnDisable()
        {
            // If Academy.Dispose has already been called, we don't need to unregister with it.
            // We don't want to even try, because this will lazily create a new Academy!
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentSendState -= SendInfo;
                Academy.Instance.DecideAction -= DecideAction;
                Academy.Instance.AgentAct -= AgentStep;
                Academy.Instance.AgentForceReset -= _AgentReset;
            }
            NotifyAgentDone();
            m_Brain?.Dispose();
            m_Initialized = false;
        }

        void NotifyAgentDone(bool maxStepReached = false)
        {
            m_Info.reward = m_Reward;
            m_Info.done = true;
            m_Info.maxStepReached = maxStepReached;
            // Request the last decision with no callbacks
            // We request a decision so Python knows the Agent is done immediately
            m_Brain?.RequestDecision(m_Info, sensors);

            UpdateRewardStats();

            // The Agent is done, so we give it a new episode Id
            m_EpisodeId = EpisodeIdCounter.GetEpisodeId();
            m_Reward = 0f;
            m_CumulativeReward = 0f;
            m_RequestAction = false;
            m_RequestDecision = false;
        }

        /// <summary>
        /// Updates the Model for the agent. Any model currently assigned to the
        /// agent will be replaced with the provided one. If the arguments are
        /// identical to the current parameters of the agent, the model will
        /// remain unchanged.
        /// </summary>
        /// <param name="behaviorName"> The identifier of the behavior. This
        /// will categorize the agent when training.
        /// </param>
        /// <param name="model"> The model to use for inference.</param>
        /// <param name = "inferenceDevice"> Define on what device the model
        /// will be run.</param>
        public void GiveModel(
            string behaviorName,
            NNModel model,
            InferenceDevice inferenceDevice = InferenceDevice.CPU)
        {
            m_PolicyFactory.GiveModel(behaviorName, model, inferenceDevice);
            m_Brain?.Dispose();
            m_Brain = m_PolicyFactory.GeneratePolicy(Heuristic);
        }

        /// <summary>
        /// Returns the current step counter (within the current episode).
        /// </summary>
        /// <returns>
        /// Current episode number.
        /// </returns>
        public int GetStepCount()
        {
            return m_StepCount;
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
#if DEBUG
            if (float.IsNaN(reward))
            {
                throw new ArgumentException("NaN reward passed to SetReward.");
            }
#endif
            m_CumulativeReward += (reward - m_Reward);
            m_Reward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
#if DEBUG
            if (float.IsNaN(increment))
            {
                throw new ArgumentException("NaN reward passed to AddReward.");
            }
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
        /// Sets the done flag to true.
        /// </summary>
        public void Done()
        {
            NotifyAgentDone();
            _AgentReset();
        }

        /// <summary>
        /// Is called when the agent must request the brain for a new decision.
        /// </summary>
        public void RequestDecision()
        {
            m_RequestDecision = true;
            RequestAction();
        }

        /// <summary>
        /// Is called then the agent must perform a new action.
        /// </summary>
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
            m_ActionMasker = new ActionMasker(param);
            // If we haven't initialized vectorActions, initialize to 0. This should only
            // happen during the creation of the Agent. In subsequent episodes, vectorAction
            // should stay the previous action before the Done(), so that it is properly recorded.
            if (m_Action.vectorActions == null)
            {
                if (param.vectorActionSpaceType == SpaceType.Continuous)
                {
                    m_Action.vectorActions = new float[param.vectorActionSize[0]];
                    m_Info.storedVectorActions = new float[param.vectorActionSize[0]];
                }
                else
                {
                    m_Action.vectorActions = new float[param.vectorActionSize.Length];
                    m_Info.storedVectorActions = new float[param.vectorActionSize.Length];
                }
            }
        }

        /// <summary>
        /// Initializes the agent, called once when the agent is enabled. Can be
        /// left empty if there is no special, unique set-up behavior for the
        /// agent.
        /// </summary>
        /// <remarks>
        /// One sample use is to store local references to other objects in the
        /// scene which would facilitate computing this agents observation.
        /// </remarks>
        public virtual void InitializeAgent()
        {
        }

        /// <summary>
        /// When the Agent uses Heuristics, it will call this method every time it
        /// needs an action. This can be used for debugging or controlling the agent
        /// with keyboard.
        /// </summary>
        /// <returns> A float array corresponding to the next action of the Agent
        /// </returns>
        public virtual float[] Heuristic()
        {
            throw new UnityAgentsException(string.Format(
                "The Heuristic method was not implemented for the Agent on the " +
                "{0} GameObject.",
                gameObject.name));
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
                    var stackingSensor = new StackingSensor(collectObservationsSensor, param.numStackedVectorObservations);
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
                Debug.Assert(!sensors[i].GetName().Equals(sensors[i + 1].GetName()), "Sensor names must be unique.");
            }
#endif
        }

        /// <summary>
        /// Sends the Agent info to the linked Brain.
        /// </summary>
        void SendInfoToBrain()
        {
            if (m_Brain == null)
            {
                return;
            }

            m_Info.storedVectorActions = m_Action.vectorActions;
            m_ActionMasker.ResetMask();
            UpdateSensors();
            using (TimerStack.Instance.Scoped("CollectObservations"))
            {
                CollectObservations(collectObservationsSensor, m_ActionMasker);
            }
            m_Info.actionMasks = m_ActionMasker.GetMask();

            m_Info.reward = m_Reward;
            m_Info.done = false;
            m_Info.maxStepReached = false;
            m_Info.episodeId = m_EpisodeId;

            m_Brain.RequestDecision(m_Info, sensors);

            if (m_Recorder != null && m_Recorder.record && Application.isEditor)
            {
                m_Recorder.WriteExperience(m_Info, sensors);
            }
        }

        void UpdateSensors()
        {
            for (var i = 0; i < sensors.Count; i++)
            {
                sensors[i].Update();
            }
        }

        /// <summary>
        /// Collects the vector observations of the agent.
        /// The agent observation describes the current environment from the
        /// perspective of the agent.
        /// </summary>
        /// <remarks>
        /// An agents observation is any environment information that helps
        /// the Agent achieve its goal. For example, for a fighting Agent, its
        /// observation could include distances to friends or enemies, or the
        /// current level of ammunition at its disposal.
        /// Recall that an Agent may attach vector or visual observations.
        /// Vector observations are added by calling the provided helper methods
        /// on the VectorSensor input:
        ///     - <see cref="AddObservation(int)"/>
        ///     - <see cref="AddObservation(float)"/>
        ///     - <see cref="AddObservation(Vector3)"/>
        ///     - <see cref="AddObservation(Vector2)"/>
        ///     - <see cref="AddObservation(Quaternion)"/>
        ///     - <see cref="AddObservation(bool)"/>
        ///     - <see cref="AddOneHotObservation(int, int)"/>
        /// Depending on your environment, any combination of these helpers can
        /// be used. They just need to be used in the exact same order each time
        /// this method is called and the resulting size of the vector observation
        /// needs to match the vectorObservationSize attribute of the linked Brain.
        /// Visual observations are implicitly added from the cameras attached to
        /// the Agent.
        /// </remarks>
        public virtual void CollectObservations(VectorSensor sensor)
        {
        }

        /// <summary>
        /// Collects the vector observations of the agent.
        /// The agent observation describes the current environment from the
        /// perspective of the agent.
        /// </summary>
        /// <remarks>
        /// An agents observation is any environment information that helps
        /// the Agent achieve its goal. For example, for a fighting Agent, its
        /// observation could include distances to friends or enemies, or the
        /// current level of ammunition at its disposal.
        /// Recall that an Agent may attach vector or visual observations.
        /// Vector observations are added by calling the provided helper methods
        /// on the VectorSensor input:
        ///     - <see cref="AddObservation(int)"/>
        ///     - <see cref="AddObservation(float)"/>
        ///     - <see cref="AddObservation(Vector3)"/>
        ///     - <see cref="AddObservation(Vector2)"/>
        ///     - <see cref="AddObservation(Quaternion)"/>
        ///     - <see cref="AddObservation(bool)"/>
        ///     - <see cref="AddOneHotObservation(int, int)"/>
        /// Depending on your environment, any combination of these helpers can
        /// be used. They just need to be used in the exact same order each time
        /// this method is called and the resulting size of the vector observation
        /// needs to match the vectorObservationSize attribute of the linked Brain.
        /// Visual observations are implicitly added from the cameras attached to
        /// the Agent.
        /// When using Discrete Control, you can prevent the Agent from using a certain
        /// action by masking it. You can call the following method on the ActionMasker
        /// input :
        ///     - <see cref="SetActionMask(int branch, IEnumerable<int> actionIndices)"/>
        ///     - <see cref="SetActionMask(int branch, int actionIndex)"/>
        ///     - <see cref="SetActionMask(IEnumerable<int> actionIndices)"/>
        ///     - <see cref="SetActionMask(int branch, int actionIndex)"/>
        /// The branch input is the index of the action, actionIndices are the indices of the
        /// invalid options for that action.
        /// </remarks>
        public virtual void CollectObservations(VectorSensor sensor, ActionMasker actionMasker)
        {
            CollectObservations(sensor);
        }

        /// <summary>
        /// Specifies the agent behavior at every step based on the provided
        /// action.
        /// </summary>
        /// <param name="vectorAction">
        /// Vector action. Note that for discrete actions, the provided array
        /// will be of length 1.
        /// </param>
        public virtual void AgentAction(float[] vectorAction)
        {
        }

        /// <summary>
        /// Specifies the agent behavior when being reset, which can be due to
        /// the agent or Academy being done (i.e. completion of local or global
        /// episode).
        /// </summary>
        public virtual void AgentReset()
        {
        }

        /// <summary>
        /// Returns the last action that was decided on by the Agent (returns null if no decision has been made)
        /// </summary>
        public float[] GetAction()
        {
            return m_Action.vectorActions;
        }

        /// <summary>
        /// This method will forcefully reset the agent and will also reset the hasAlreadyReset flag.
        /// This way, even if the agent was already in the process of reseting, it will be reset again
        /// and will not send a Done flag at the next step.
        /// </summary>
        void ForceReset()
        {
            _AgentReset();
        }

        /// <summary>
        /// An internal reset method that updates internal data structures in
        /// addition to calling <see cref="AgentReset"/>.
        /// </summary>
        void _AgentReset()
        {
            ResetData();
            m_StepCount = 0;
            AgentReset();
        }

        /// <summary>
        /// Scales continuous action from [-1, 1] to arbitrary range.
        /// </summary>
        /// <param name="rawAction"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        protected float ScaleAction(float rawAction, float min, float max)
        {
            var middle = (min + max) / 2;
            var range = (max - min) / 2;
            return rawAction * range + middle;
        }

        /// <summary>
        /// Signals the agent that it must sent its decision to the brain.
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

        /// Used by the brain to make the agent perform a step.
        void AgentStep()
        {
            if ((m_StepCount >= maxStep) && (maxStep > 0))
            {
                NotifyAgentDone(true);
                _AgentReset();
            }
            else
            {
                m_StepCount += 1;
            }

            if ((m_RequestAction) && (m_Brain != null))
            {
                m_RequestAction = false;
                if (m_Action.vectorActions != null)
                {
                    AgentAction(m_Action.vectorActions);
                }
            }
        }

        void DecideAction()
        {
            m_Action.vectorActions = m_Brain?.DecideAction();
        }
    }
}
