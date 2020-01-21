using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using MLAgents.Sensor;
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
        /// Unique identifier each agent receives at initialization. It is used
        /// to separate between different agents in the environment.
        /// </summary>
        public int id;
    }

    /// <summary>
    /// Struct that contains the action information sent from the Brain to the
    /// Agent.
    /// </summary>
    public struct AgentAction
    {
        public float[] vectorActions;
        public float value;
    }

    /// <summary>
    /// Struct that contains all the Agent-specific parameters provided in the
    /// Editor. This excludes the Brain linked to the Agent since it can be
    /// modified programmatically.
    /// </summary>
    [Serializable]
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
    public abstract class Agent : MonoBehaviour
    {
        IPolicy m_Brain;
        BehaviorParameters m_PolicyFactory;

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

        /// Unique identifier each agent receives at initialization. It is used
        /// to separate between different agents in the environment.
        int m_Id;

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
        [FormerlySerializedAs("m_Sensors")]
        public List<ISensor> sensors;

        /// <summary>
        /// VectorSensor which is written to by AddVectorObs
        /// </summary>
        public VectorSensor collectObservationsSensor;

        /// MonoBehaviour function that is called when the attached GameObject
        /// becomes enabled or active.
        void OnEnable()
        {
            m_Id = gameObject.GetInstanceID();
            OnEnableHelper();

            m_Recorder = GetComponent<DemonstrationRecorder>();
        }

        /// Helper method for the <see cref="OnEnable"/> event, created to
        /// facilitate testing.
        void OnEnableHelper()
        {
            m_Info = new AgentInfo();
            m_Action = new AgentAction();
            sensors = new List<ISensor>();

            Academy.Instance.AgentResetIfDone += ResetIfDone;
            Academy.Instance.AgentSendState += SendInfo;
            Academy.Instance.DecideAction += DecideAction;
            Academy.Instance.AgentAct += AgentStep;
            Academy.Instance.AgentForceReset += _AgentReset;
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
            // If Academy.Dispose has already been called, we don't need to unregister with it.
            // We don't want to even try, because this will lazily create a new Academy!
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentResetIfDone -= ResetIfDone;
                Academy.Instance.AgentSendState -= SendInfo;
                Academy.Instance.DecideAction -= DecideAction;
                Academy.Instance.AgentAct -= AgentStep;
                Academy.Instance.AgentForceReset -= _AgentReset;
            }
            NotifyAgentDone();
            m_Brain?.Dispose();
        }

        void NotifyAgentDone()
        {
            m_Info.done = true;
            // Request the last decision with no callbacks
            // We request a decision so Python knows the Agent is disabled
            m_Brain?.RequestDecision(m_Info, sensors, (a) => { });
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
                CollectObservations();
            }
            m_Info.actionMasks = m_ActionMasker.GetMask();

            m_Info.reward = m_Reward;
            m_Info.done = m_Done;
            m_Info.maxStepReached = m_MaxStepReached;
            m_Info.id = m_Id;

            m_Brain.RequestDecision(m_Info, sensors, UpdateAgentAction);

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
        /// Collects the (vector, visual) observations of the agent.
        /// The agent observation describes the current environment from the
        /// perspective of the agent.
        /// </summary>
        /// <remarks>
        /// Simply, an agents observation is any environment information that helps
        /// the Agent acheive its goal. For example, for a fighting Agent, its
        /// observation could include distances to friends or enemies, or the
        /// current level of ammunition at its disposal.
        /// Recall that an Agent may attach vector or visual observations.
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
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds an integer observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 1.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(int observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds an Vector3 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 3.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector3 observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds an Vector2 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 2.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector2 observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds a collection of float observations to the vector observations of the agent.
        /// Increases the size of the agents vector observation by size of the collection.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(IEnumerable<float> observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds a quaternion observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 4.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Quaternion observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        /// <summary>
        /// Adds a boolean observation to the vector observation of the agent.
        /// Increases the size of the agent's vector observation by 1.
        /// </summary>
        /// <param name="observation"></param>
        protected void AddVectorObs(bool observation)
        {
            collectObservationsSensor.AddObservation(observation);
        }

        protected void AddVectorObs(int observation, int range)
        {
            collectObservationsSensor.AddOneHotObservation(observation, range);
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


        /// Signals the agent that it must reset if its done flag is set to true.
        void ResetIfDone()
        {
            if (IsDone())
            {
                _AgentReset();
            }
        }

        /// <summary>
        /// Signals the agent that it must sent its decision to the brain.
        /// </summary>
        void SendInfo()
        {
            // If the Agent is done, it has just reset and thus requires a new decision
            if (m_RequestDecision || IsDone())
            {
                SendInfoToBrain();
                ResetReward();
                m_Done = false;
                m_MaxStepReached = false;
                m_RequestDecision = false;
            }
        }

        /// Used by the brain to make the agent perform a step.
        void AgentStep()
        {
            if ((m_RequestAction) && (m_Brain != null))
            {
                m_RequestAction = false;
                AgentAction(m_Action.vectorActions);
            }

            if ((m_StepCount >= agentParameters.maxStep)
                && (agentParameters.maxStep > 0))
            {
                m_MaxStepReached = true;
                Done();
            }

            m_StepCount += 1;
        }

        void DecideAction()
        {
            m_Brain?.DecideAction();
        }
    }
}
