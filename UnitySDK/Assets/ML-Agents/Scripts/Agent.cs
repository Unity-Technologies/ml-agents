using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using MLAgents.Sensor;



namespace MLAgents
{
    /// <summary>
    /// Struct that contains all the information for an Agent, including its
    /// observations, actions and current status, that is sent to the Brain.
    /// </summary>
    public struct AgentInfo
    {
        /// <summary>
        /// Most recent agent vector (i.e. numeric) observation.
        /// </summary>
        public List<float> vectorObservation;

        /// <summary>
        /// The previous agent vector observations, stacked. The length of the
        /// history (i.e. number of vector observations to stack) is specified
        /// in the Brain parameters.
        /// </summary>
        public List<float> stackedVectorObservation;

        /// <summary>
        /// Most recent compressed observations.
        /// </summary>
        public List<CompressedObservation> compressedObservations;

        /// <summary>
        /// Most recent text observation.
        /// </summary>
        public string textObservation;

        /// <summary>
        /// Keeps track of the last vector action taken by the Brain.
        /// </summary>
        public float[] storedVectorActions;

        /// <summary>
        /// Keeps track of the last text action taken by the Brain.
        /// </summary>
        public string storedTextActions;

        /// <summary>
        /// For discrete control, specifies the actions that the agent cannot take. Is true if
        /// the action is masked.
        /// </summary>
        public bool[] actionMasks;

        /// <summary>
        /// Used by the Trainer to store information about the agent. This data
        /// structure is not consumed or modified by the agent directly, they are
        /// just the owners of their trainier's memory. Currently, however, the
        /// size of the memory is in the Brain properties.
        /// </summary>
        public List<float> memories;

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
        /// Unique identifier each agent receives at initialization. It is used
        /// to separate between different agents in the environment.
        /// </summary>
        public int id;

        /// <summary>
        /// User-customizable object for sending structured output from Unity to Python in response
        /// to an action in addition to a scalar reward.
        /// TODO(cgoy): All references to protobuf objects should be removed.
        /// </summary>
        public CommunicatorObjects.CustomObservationProto customObservation;
    }

    /// <summary>
    /// Struct that contains the action information sent from the Brain to the
    /// Agent.
    /// </summary>
    public struct AgentAction
    {
        public float[] vectorActions;
        public string textActions;
        public List<float> memories;
        public float value;
        /// TODO(cgoy): All references to protobuf objects should be removed.
        public CommunicatorObjects.CustomActionProto customAction;
    }

    /// <summary>
    /// Struct that contains all the Agent-specific parameters provided in the
    /// Editor. This excludes the Brain linked to the Agent since it can be
    /// modified programmatically.
    /// </summary>
    [System.Serializable]
    public class AgentParameters
    {
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
        public int maxStep;

        /// <summary>
        /// Determines the behaviour of the agent when done.
        /// </summary>
        /// <remarks>
        /// If true, the agent will reset when done and start a new episode.
        /// Otherwise, the agent will remain done and its behavior will be
        /// dictated by the AgentOnDone method.
        /// </remarks>
        public bool resetOnDone = true;

        /// <summary>
        /// Whether to enable On Demand Decisions or make a decision at
        /// every step.
        /// </summary>
        public bool onDemandDecision;

        /// <summary>
        /// Number of actions between decisions (used when On Demand Decisions
        /// is turned off).
        /// </summary>
        public int numberOfActionsBetweenDecisions;
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
    [System.Serializable]
    [RequireComponent(typeof(BehaviorParameters))]
    public abstract class Agent : MonoBehaviour
    {
        private IPolicy m_Brain;
        private BehaviorParameters m_PolicyFactory;

        /// <summary>
        /// Agent parameters specified within the Editor via AgentEditor.
        /// </summary>
        [HideInInspector] public AgentParameters agentParameters;

        /// Current Agent information (message sent to Brain).
        AgentInfo m_Info;
        public AgentInfo Info
        {
            get { return m_Info; }
            set { m_Info = value; }
        }

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

        /// Whether or not the agent has completed the episode. This may be due
        /// to either reaching a success or fail state, or reaching the maximum
        /// number of steps (i.e. timing out).
        bool m_Done;

        /// Whether or not the agent reached the maximum number of steps.
        bool m_MaxStepReached;

        /// Keeps track of the number of steps taken by the agent in this episode.
        /// Note that this value is different for each agent, and may not overlap
        /// with the step counter in the Academy, since agents reset based on
        /// their own experience.
        int m_StepCount;

        /// Flag to signify that an agent has been reset but the fact that it is
        /// done has not been communicated (required for On Demand Decisions).
        bool m_HasAlreadyReset;

        /// Flag to signify that an agent is done and should not reset until
        /// the fact that it is done has been communicated.
        bool m_Terminate;

        /// Unique identifier each agent receives at initialization. It is used
        /// to separate between different agents in the environment.
        int m_Id;

        /// Keeps track of the actions that are masked at each step.
        private ActionMasker m_ActionMasker;

        /// <summary>
        /// Demonstration recorder.
        /// </summary>
        private DemonstrationRecorder m_Recorder;

        public List<ISensor> m_Sensors;

        /// Monobehavior function that is called when the attached GameObject
        /// becomes enabled or active.
        void OnEnable()
        {
            m_Id = gameObject.GetInstanceID();
            var academy = FindObjectOfType<Academy>();
            academy.LazyInitialization();
            OnEnableHelper(academy);

            m_Recorder = GetComponent<DemonstrationRecorder>();
        }

        /// Helper method for the <see cref="OnEnable"/> event, created to
        /// facilitate testing.
        void OnEnableHelper(Academy academy)
        {
            m_Info = new AgentInfo();
            m_Action = new AgentAction();
            m_Sensors = new List<ISensor>();

            if (academy == null)
            {
                throw new UnityAgentsException(
                    "No Academy Component could be found in the scene.");
            }

            academy.AgentSetStatus += SetStatus;
            academy.AgentResetIfDone += ResetIfDone;
            academy.AgentSendState += SendInfo;
            academy.DecideAction += DecideAction;
            academy.AgentAct += AgentStep;
            academy.AgentForceReset += _AgentReset;
            m_PolicyFactory = GetComponent<BehaviorParameters>();
            m_Brain = m_PolicyFactory.GeneratePolicy(Heuristic);
            ResetData();
            InitializeAgent();
            InitializeSensors();
        }

        /// Monobehavior function that is called when the attached GameObject
        /// becomes disabled or inactive.
        void OnDisable()
        {
            var academy = FindObjectOfType<Academy>();
            if (academy != null)
            {
                academy.AgentSetStatus -= SetStatus;
                academy.AgentResetIfDone -= ResetIfDone;
                academy.AgentSendState -= SendInfo;
                academy.DecideAction -= DecideAction;
                academy.AgentAct -= AgentStep;
                academy.AgentForceReset -= ForceReset;
            }
            m_Brain?.Dispose();
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
        /// <param name = "inferenceDevide"> Define on what device the model
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
        /// Returns the current step counter (within the current epside).
        /// </summary>
        /// <returns>
        /// Current episode number.
        /// </returns>
        public int GetStepCount()
        {
            return m_StepCount;
        }

        /// <summary>
        /// Resets the step reward and possibly the episode reward for the agent.
        /// </summary>
        public void ResetReward()
        {
            m_Reward = 0f;
            if (m_Done)
            {
                m_CumulativeReward = 0f;
            }
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
            m_CumulativeReward += (reward - m_Reward);
            m_Reward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
            m_Reward += increment;
            m_CumulativeReward += increment;
        }

        /// <summary>
        /// Retrieves the step reward for the Agent.
        /// </summary>
        /// <returns>The step reward.</returns>
        public float GetReward()
        {
            return m_Reward;
        }

        /// <summary>
        /// Retrieves the episode reward for the Agent.
        /// </summary>
        /// <returns>The episode reward.</returns>
        public float GetCumulativeReward()
        {
            return m_CumulativeReward;
        }

        /// <summary>
        /// Sets the done flag to true.
        /// </summary>
        public void Done()
        {
            m_Done = true;
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

        /// <summary>
        /// Indicates if the agent has reached his maximum number of steps.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if max step reached was reached, <c>false</c> otherwise.
        /// </returns>
        public bool IsMaxStepReached()
        {
            return m_MaxStepReached;
        }

        /// <summary>
        /// Indicates if the agent is done
        /// </summary>
        /// <returns>
        /// <c>true</c>, if the agent is done, <c>false</c> otherwise.
        /// </returns>
        public bool IsDone()
        {
            return m_Done;
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

            if (m_Info.textObservation == null)
                m_Info.textObservation = "";
            m_Action.textActions = "";
            m_Info.memories = new List<float>();
            m_Action.memories = new List<float>();
            m_Info.vectorObservation =
                new List<float>(param.vectorObservationSize);
            m_Info.stackedVectorObservation =
                new List<float>(param.vectorObservationSize
                    * param.numStackedVectorObservations);
            m_Info.stackedVectorObservation.AddRange(
                new float[param.vectorObservationSize
                          * param.numStackedVectorObservations]);

            m_Info.compressedObservations = new List<CompressedObservation>();
            m_Info.customObservation = null;
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
        public void InitializeSensors()
        {
            var attachedSensorComponents = GetComponents<SensorComponent>();
            m_Sensors.Capacity += attachedSensorComponents.Length;
            foreach (var component in attachedSensorComponents)
            {
                m_Sensors.Add(component.CreateSensor());
            }

            // Sort the sensors by name to ensure determinism
            m_Sensors.Sort((x, y) => x.GetName().CompareTo(y.GetName()));

#if DEBUG
            // Make sure the names are actually unique
            for (var i = 0; i < m_Sensors.Count - 1; i++)
            {
                Debug.Assert(!m_Sensors[i].GetName().Equals(m_Sensors[i + 1].GetName()), "Sensor names must be unique.");
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

            m_Info.memories = m_Action.memories;
            m_Info.storedVectorActions = m_Action.vectorActions;
            m_Info.storedTextActions = m_Action.textActions;
            m_Info.vectorObservation.Clear();
            m_Info.compressedObservations.Clear();
            m_ActionMasker.ResetMask();
            using (TimerStack.Instance.Scoped("CollectObservations"))
            {
                CollectObservations();
            }
            m_Info.actionMasks = m_ActionMasker.GetMask();

            var param = m_PolicyFactory.brainParameters;
            if (m_Info.vectorObservation.Count != param.vectorObservationSize)
            {
                throw new UnityAgentsException(string.Format(
                    "Vector Observation size mismatch in continuous " +
                    "agent {0}. " +
                    "Was Expecting {1} but received {2}. ",
                    gameObject.name,
                    param.vectorObservationSize,
                    m_Info.vectorObservation.Count));
            }

            Utilities.ShiftLeft(m_Info.stackedVectorObservation, param.vectorObservationSize);
            Utilities.ReplaceRange(m_Info.stackedVectorObservation, m_Info.vectorObservation,
                m_Info.stackedVectorObservation.Count - m_Info.vectorObservation.Count);

            m_Info.reward = m_Reward;
            m_Info.done = m_Done;
            m_Info.maxStepReached = m_MaxStepReached;
            m_Info.id = m_Id;

            m_Brain.RequestDecision(this);

            if (m_Recorder != null && m_Recorder.record && Application.isEditor)
            {
                // This is a bit of a hack - if we're in inference mode, compressed observations won't be generated
                // But we need these to be generated for the recorder. So generate them here.
                if (m_Info.compressedObservations.Count == 0)
                {
                    GenerateSensorData();
                }

                m_Recorder.WriteExperience(m_Info);
            }

            m_Info.textObservation = "";
        }

        /// <summary>
        /// Generate data for each sensor and store it on the Agent's AgentInfo.
        /// NOTE: At the moment, this is only called during training or when using a DemonstrationRecorder;
        /// during inference the sensors are used to write directly to the Tensor data. This will likely change in the
        /// future to be controlled by the type of brain being used.
        /// </summary>
        public void GenerateSensorData()
        {
            // Generate data for all sensors
            // TODO add bool argument indicating when to compress? For now, we always will compress.
            for (var i = 0; i < m_Sensors.Count; i++)
            {
                var sensor = m_Sensors[i];
                var compressedObs = new CompressedObservation
                {
                    Data = sensor.GetCompressedObservation(),
                    Shape = sensor.GetFloatObservationShape(),
                    CompressionType = sensor.GetCompressionType()
                };
                m_Info.compressedObservations.Add(compressedObs);
            }
        }

        /// <summary>
        /// Collects the (vector, visual, text) observations of the agent.
        /// The agent observation describes the current environment from the
        /// perspective of the agent.
        /// </summary>
        /// <remarks>
        /// Simply, an agents observation is any environment information that helps
        /// the Agent acheive its goal. For example, for a fighting Agent, its
        /// observation could include distances to friends or enemies, or the
        /// current level of ammunition at its disposal.
        /// Recall that an Agent may attach vector, visual or textual observations.
        /// Vector observations are added by calling the provided helper methods:
        ///     - <see cref="AddVectorObs(int)"/>
        ///     - <see cref="AddVectorObs(float)"/>
        ///     - <see cref="AddVectorObs(Vector3)"/>
        ///     - <see cref="AddVectorObs(Vector2)"/>
        ///     - <see>
        ///         <cref>AddVectorObs(float[])</cref>
        ///       </see>
        ///     - <see>
        ///         <cref>AddVectorObs(List{float})</cref>
        ///      </see>
        ///     - <see cref="AddVectorObs(Quaternion)"/>
        ///     - <see cref="AddVectorObs(bool)"/>
        ///     - <see cref="AddVectorObs(int, int)"/>
        /// Depending on your environment, any combination of these helpers can
        /// be used. They just need to be used in the exact same order each time
        /// this method is called and the resulting size of the vector observation
        /// needs to match the vectorObservationSize attribute of the linked Brain.
        /// Visual observations are implicitly added from the cameras attached to
        /// the Agent.
        /// Lastly, textual observations are added using
        /// <see cref="SetTextObs(string)"/>.
        /// </remarks>
        public virtual void CollectObservations()
        {
        }

        /// <summary>
        /// Sets an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the action passed as argument at the next decision. If no branch is
        /// specified, the default branch will be 0. The actionIndex or actionIndices correspond
        /// to the action the agent will be unable to perform.
        /// </summary>
        /// <param name="actionIndices">The indices of the masked actions on branch 0</param>
        protected void SetActionMask(IEnumerable<int> actionIndices)
        {
            m_ActionMasker.SetActionMask(0, actionIndices);
        }

        /// <summary>
        /// Sets an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the action passed as argument at the next decision. If no branch is
        /// specified, the default branch will be 0. The actionIndex or actionIndices correspond
        /// to the action the agent will be unable to perform.
        /// </summary>
        /// <param name="actionIndex">The index of the masked action on branch 0</param>
        protected void SetActionMask(int actionIndex)
        {
            m_ActionMasker.SetActionMask(0, new[] { actionIndex });
        }

        /// <summary>
        /// Sets an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the action passed as argument at the next decision. If no branch is
        /// specified, the default branch will be 0. The actionIndex or actionIndices correspond
        /// to the action the agent will be unable to perform.
        /// </summary>
        /// <param name="branch">The branch for which the actions will be masked</param>
        /// <param name="actionIndex">The index of the masked action</param>
        protected void SetActionMask(int branch, int actionIndex)
        {
            m_ActionMasker.SetActionMask(branch, new[] { actionIndex });
        }

        /// <summary>
        /// Modifies an action mask for discrete control agents. When used, the agent will not be
        /// able to perform the action passed as argument at the next decision. If no branch is
        /// specified, the default branch will be 0. The actionIndex or actionIndices correspond
        /// to the action the agent will be unable to perform.
        /// </summary>
        /// <param name="branch">The branch for which the actions will be masked</param>
        /// <param name="actionIndices">The indices of the masked actions</param>
        protected void SetActionMask(int branch, IEnumerable<int> actionIndices)
        {
            m_ActionMasker.SetActionMask(branch, actionIndices);
        }

        /// <summary>
        /// Adds a float observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 1.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(float observation)
        {
            m_Info.vectorObservation.Add(observation);
        }

        /// <summary>
        /// Adds an integer observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 1.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(int observation)
        {
            m_Info.vectorObservation.Add(observation);
        }

        /// <summary>
        /// Adds an Vector3 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 3.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector3 observation)
        {
            m_Info.vectorObservation.Add(observation.x);
            m_Info.vectorObservation.Add(observation.y);
            m_Info.vectorObservation.Add(observation.z);
        }

        /// <summary>
        /// Adds an Vector2 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 2.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector2 observation)
        {
            m_Info.vectorObservation.Add(observation.x);
            m_Info.vectorObservation.Add(observation.y);
        }

        /// <summary>
        /// Adds a collection of float observations to the vector observations of the agent.
        /// Increases the size of the agents vector observation by size of the collection.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(IEnumerable<float> observation)
        {
            m_Info.vectorObservation.AddRange(observation);
        }

        /// <summary>
        /// Adds a quaternion observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 4.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Quaternion observation)
        {
            m_Info.vectorObservation.Add(observation.x);
            m_Info.vectorObservation.Add(observation.y);
            m_Info.vectorObservation.Add(observation.z);
            m_Info.vectorObservation.Add(observation.w);
        }

        /// <summary>
        /// Adds a boolean observation to the vector observation of the agent.
        /// Increases the size of the agent's vector observation by 1.
        /// </summary>
        /// <param name="observation"></param>
        protected void AddVectorObs(bool observation)
        {
            m_Info.vectorObservation.Add(observation ? 1f : 0f);
        }

        protected void AddVectorObs(int observation, int range)
        {
            var oneHotVector = new float[range];
            oneHotVector[observation] = 1;
            m_Info.vectorObservation.AddRange(oneHotVector);
        }

        /// <summary>
        /// Sets the text observation.
        /// </summary>
        /// <param name="textObservation">The text observation.</param>
        public void SetTextObs(string textObservation)
        {
            m_Info.textObservation = textObservation;
        }

        /// <summary>
        /// Specifies the agent behavior at every step based on the provided
        /// action.
        /// </summary>
        /// <param name="vectorAction">
        /// Vector action. Note that for discrete actions, the provided array
        /// will be of length 1.
        /// </param>
        /// <param name="textAction">Text action.</param>
        public virtual void AgentAction(float[] vectorAction, string textAction)
        {
        }

        /// <summary>
        /// Specifies the agent behavior at every step based on the provided
        /// action.
        /// </summary>
        /// <param name="vectorAction">
        /// Vector action. Note that for discrete actions, the provided array
        /// will be of length 1.
        /// </param>
        /// <param name="textAction">Text action.</param>
        /// <param name="customAction">
        /// A custom action, defined by the user as custom protobuf message. Useful if the action is hard to encode
        /// as either a flat vector or a single string.
        /// </param>
        public virtual void AgentAction(float[] vectorAction, string textAction, CommunicatorObjects.CustomActionProto customAction)
        {
            // We fall back to not using the custom action if the subclassed Agent doesn't override this method.
            AgentAction(vectorAction, textAction);
        }

        /// <summary>
        /// Specifies the agent behavior when done and
        /// <see cref="AgentParameters.resetOnDone"/> is false. This method can be
        /// used to remove the agent from the scene.
        /// </summary>
        public virtual void AgentOnDone()
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
        /// This method will forcefully reset the agent and will also reset the hasAlreadyReset flag.
        /// This way, even if the agent was already in the process of reseting, it will be reset again
        /// and will not send a Done flag at the next step.
        /// </summary>
        void ForceReset()
        {
            m_HasAlreadyReset = false;
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

        public void UpdateAgentAction(AgentAction action)
        {
            m_Action = action;
        }

        /// <summary>
        /// Updates the vector action.
        /// </summary>
        /// <param name="vectorActions">Vector actions.</param>
        public void UpdateVectorAction(float[] vectorActions)
        {
            m_Action.vectorActions = vectorActions;
        }

        /// <summary>
        /// Updates the memories action.
        /// </summary>
        /// <param name="memories">Memories.</param>
        public void UpdateMemoriesAction(List<float> memories)
        {
            m_Action.memories = memories;
        }

        public void AppendMemoriesAction(List<float> memories)
        {
            m_Action.memories.AddRange(memories);
        }

        public List<float> GetMemoriesAction()
        {
            return m_Action.memories;
        }

        /// <summary>
        /// Updates the value of the agent.
        /// </summary>
        public void UpdateValueAction(float value)
        {
            m_Action.value = value;
        }

        protected float GetValueEstimate()
        {
            return m_Action.value;
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
        /// Sets the status of the agent. Will request decisions or actions according
        /// to the Academy's stepcount.
        /// </summary>
        /// <param name="academyStepCounter">Number of current steps in episode</param>
        void SetStatus(int academyStepCounter)
        {
            MakeRequests(academyStepCounter);
        }

        /// Signals the agent that it must reset if its done flag is set to true.
        void ResetIfDone()
        {
            // If an agent is done, then it will also
            // request for a decision and an action
            if (IsDone())
            {
                if (agentParameters.resetOnDone)
                {
                    if (agentParameters.onDemandDecision)
                    {
                        if (!m_HasAlreadyReset)
                        {
                            // If event based, the agent can reset as soon
                            // as it is done
                            _AgentReset();
                            m_HasAlreadyReset = true;
                        }
                    }
                    else if (m_RequestDecision)
                    {
                        // If not event based, the agent must wait to request a
                        // decision before resetting to keep multiple agents in sync.
                        _AgentReset();
                    }
                }
                else
                {
                    m_Terminate = true;
                    RequestDecision();
                }
            }
        }

        /// <summary>
        /// Signals the agent that it must sent its decision to the brain.
        /// </summary>
        void SendInfo()
        {
            if (m_RequestDecision)
            {
                SendInfoToBrain();
                ResetReward();
                m_Done = false;
                m_MaxStepReached = false;
                m_RequestDecision = false;

                m_HasAlreadyReset = false;
            }
        }

        /// Used by the brain to make the agent perform a step.
        void AgentStep()
        {
            if (m_Terminate)
            {
                m_Terminate = false;
                ResetReward();
                m_Done = false;
                m_MaxStepReached = false;
                m_RequestDecision = false;
                m_RequestAction = false;

                m_HasAlreadyReset = false;
                OnDisable();
                AgentOnDone();
            }

            if ((m_RequestAction) && (m_Brain != null))
            {
                m_RequestAction = false;
                AgentAction(m_Action.vectorActions, m_Action.textActions, m_Action.customAction);
            }

            if ((m_StepCount >= agentParameters.maxStep)
                && (agentParameters.maxStep > 0))
            {
                m_MaxStepReached = true;
                Done();
            }

            m_StepCount += 1;
        }

        /// <summary>
        /// Is called after every step, contains the logic to decide if the agent
        /// will request a decision at the next step.
        /// </summary>
        void MakeRequests(int academyStepCounter)
        {
            agentParameters.numberOfActionsBetweenDecisions =
                Mathf.Max(agentParameters.numberOfActionsBetweenDecisions, 1);
            if (!agentParameters.onDemandDecision)
            {
                RequestAction();
                if (academyStepCounter %
                    agentParameters.numberOfActionsBetweenDecisions == 0)
                {
                    RequestDecision();
                }
            }
        }

        void DecideAction()
        {
            m_Brain?.DecideAction();
        }

        /// <summary>
        /// Sets the custom observation for the agent for this episode.
        /// </summary>
        /// <param name="customObservation">New value of the agent's custom observation.</param>
        public void SetCustomObservation(CommunicatorObjects.CustomObservationProto customObservation)
        {
            m_Info.customObservation = customObservation;
        }
    }
}
