using System.Collections.Generic;
using System.Linq;
using Google.Protobuf;
using MLAgents.CommunicatorObjects;
using UnityEngine;


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
        /// Most recent agent camera (i.e. texture) observation.
        /// </summary>
        public List<Texture2D> visualObservations;

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
        /// Converts a AgentInfo to a protobuffer generated AgentInfoProto
        /// </summary>
        /// <returns>The protobuf verison of the AgentInfo.</returns>
        /// <param name="info">The AgentInfo to convert.</param>
        public CommunicatorObjects.AgentInfoProto ToProto()
        {
            var agentInfoProto = new CommunicatorObjects.AgentInfoProto
            {
                StackedVectorObservation = {stackedVectorObservation},
                StoredVectorActions = {storedVectorActions},
                StoredTextActions = storedTextActions,
                TextObservation = textObservation,
                Reward = reward,
                MaxStepReached = maxStepReached,
                Done = done,
                Id = id,
            };
            if (memories != null)
            {
                agentInfoProto.Memories.Add(memories);
            }

            if (actionMasks != null)
            {
                agentInfoProto.ActionMask.AddRange(actionMasks);
            }

            foreach (Texture2D obs in visualObservations)
            {
                agentInfoProto.VisualObservations.Add(
                    ByteString.CopyFrom(obs.EncodeToPNG())
                );
            }

            return agentInfoProto;
        }
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
        /// The list of the Camera GameObjects the agent uses for visual
        /// observations.
        /// </summary>
        public List<Camera> agentCameras = new List<Camera>();

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
    /// are determined by decisions produced by a linked Brain. Currently, this
    /// class is expected to be extended to implement the desired agent behavior.
    /// </summary>
    /// <remarks>
    /// Simply speaking, an agent roams through an environment and at each step
    /// of the environment extracts its current observation, sends them to its
    /// linked brain and in return receives an action from its brain. In practice,
    /// however, an agent need not send its observation at every step since very
    /// little may have changed between sucessive steps. Currently, how often an
    /// agent updates its brain with a fresh observation is determined by the
    /// Academy. 
    /// 
    /// At any step, an agent may be considered <see cref="done"/>. 
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
    /// Lastly, note that at any step the brain linked to the agent is allowed to
    /// change programmatically with <see cref="GiveBrain"/>.
    /// 
    /// Implementation-wise, it is required that this class is extended and the
    /// virtual methods overridden. For sample implementations of agent behavior,
    /// see the Examples/ directory within this Unity project.
    /// </remarks>
    [HelpURL("https://github.com/Unity-Technologies/ml-agents/blob/master/" +
             "docs/Learning-Environment-Design-Agents.md")]
    [System.Serializable]
    public abstract class Agent : MonoBehaviour
    {
        /// <summary>
        /// The Brain attached to this agent. A brain can be attached either
        /// directly from the Editor through AgentEditor or 
        /// programmatically through <see cref="GiveBrain"/>. It is OK for an agent
        /// to not have a brain, as long as no decision is requested.
        /// </summary>
        [HideInInspector] public Brain brain;

        /// <summary>
        /// Agent parameters specified within the Editor via AgentEditor.
        /// </summary>
        [HideInInspector] public AgentParameters agentParameters;

        /// Current Agent information (message sent to Brain).
        AgentInfo info;

        /// Current Agent action (message sent from Brain).
        AgentAction action;

        /// Represents the reward the agent accumulated during the current step.
        /// It is reset to 0 at the beginning of every step.
        /// Should be set to a positive value when the agent performs a "good"
        /// action that we wish to reinforce/reward, and set to a negative value
        /// when the agent performs a "bad" action that we wish to punish/deter.
        /// Additionally, the magnitude of the reward should not exceed 1.0
        float reward;

        /// Keeps track of the cumulative reward in this episode.
        float cumulativeReward;

        /// Whether or not the agent requests an action.
        bool requestAction;

        /// Whether or not the agent requests a decision.
        bool requestDecision;

        /// Whether or not the agent has completed the episode. This may be due
        /// to either reaching a success or fail state, or reaching the maximum
        /// number of steps (i.e. timing out).
        bool done;

        /// Whether or not the agent reached the maximum number of steps.
        bool maxStepReached;

        /// Keeps track of the number of steps taken by the agent in this episode.
        /// Note that this value is different for each agent, and may not overlap
        /// with the step counter in the Academy, since agents reset based on
        /// their own experience.
        int stepCount;

        /// Flag to signify that an agent has been reset but the fact that it is
        /// done has not been communicated (required for On Demand Decisions).
        bool hasAlreadyReset;

        /// Flag to signify that an agent is done and should not reset until
        /// the fact that it is done has been communicated.
        bool terminate;

        /// Unique identifier each agent receives at initialization. It is used
        /// to separate between different agents in the environment.
        int id;

        /// Keeps track of the actions that are masked at each step.
        private ActionMasker actionMasker;

        /// Array of Texture2D used to render to from render buffer before  
        /// transforming into float tensor.
        Texture2D[] textureArray;

        /// <summary>
        /// Demonstration recorder.
        /// </summary>
        private DemonstrationRecorder recorder;

        /// Monobehavior function that is called when the attached GameObject
        /// becomes enabled or active.
        void OnEnable()
        {
            textureArray = new Texture2D[agentParameters.agentCameras.Count];
            for (int i = 0; i < agentParameters.agentCameras.Count; i++)
            {
                textureArray[i] = new Texture2D(1, 1, TextureFormat.RGB24, false);
            }

            id = gameObject.GetInstanceID();
            Academy academy = Object.FindObjectOfType<Academy>() as Academy;
            OnEnableHelper(academy);

            recorder = GetComponent<DemonstrationRecorder>();
        }

        /// Helper method for the <see cref="OnEnable"/> event, created to
        /// facilitate testing.
        void OnEnableHelper(Academy academy)
        {
            info = new AgentInfo();
            action = new AgentAction();

            if (academy == null)
            {
                throw new UnityAgentsException(
                    "No Academy Component could be found in the scene.");
            }

            academy.AgentSetStatus += SetStatus;
            academy.AgentResetIfDone += ResetIfDone;
            academy.AgentSendState += SendInfo;
            academy.AgentAct += AgentStep;
            academy.AgentForceReset += _AgentReset;

            if (brain != null)
            {
                ResetData();
            }
            else
            {
                Debug.Log(
                    string.Format(
                        "The Agent component attached to the " +
                        "GameObject {0} was initialized without a brain.",
                        gameObject.name));
            }

            InitializeAgent();
        }

        /// Monobehavior function that is called when the attached GameObject
        /// becomes disabled or inactive.
        void OnDisable()
        {
            Academy academy = Object.FindObjectOfType<Academy>() as Academy;
            if (academy != null)
            {
                academy.AgentSetStatus -= SetStatus;
                academy.AgentResetIfDone -= ResetIfDone;
                academy.AgentSendState -= SendInfo;
                academy.AgentAct -= AgentStep;
                academy.AgentForceReset -= _AgentReset;
            }
        }

        /// <summary>
        /// Updates the Brain for the agent. Any brain currently assigned to the
        /// agent will be replaced with the provided one.
        /// </summary>
        /// <remarks>
        /// The agent unsubscribes from its current brain (if it has one) and
        /// subscribes to the provided brain. This enables contextual brains, that
        /// is, updating the behaviour (hence brain) of the agent depending on
        /// the context of the game. For example, we may utilize one (wandering)
        /// brain when an agent is randomly exploring an open world, but switch
        /// to another (fighting) brain when it comes into contact with an enemy.
        /// </remarks>
        /// <param name="brain">New brain to subscribe this agent to</param>
        public void GiveBrain(Brain brain)
        {
            this.brain = brain;
            ResetData();
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
        /// Resets the step reward and possibly the episode reward for the agent.
        /// </summary>
        public void ResetReward()
        {
            reward = 0f;
            if (done)
            {
                cumulativeReward = 0f;
            }
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
            cumulativeReward += (reward - this.reward);
            this.reward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
            reward += increment;
            cumulativeReward += increment;
        }

        /// <summary>
        /// Retrieves the step reward for the Agent.
        /// </summary>
        /// <returns>The step reward.</returns>
        public float GetReward()
        {
            return reward;
        }

        /// <summary>
        /// Retrieves the episode reward for the Agent.
        /// </summary>
        /// <returns>The episode reward.</returns>
        public float GetCumulativeReward()
        {
            return cumulativeReward;
        }

        /// <summary>
        /// Sets the done flag to true.
        /// </summary>
        public void Done()
        {
            done = true;
        }

        /// <summary>
        /// Is called when the agent must request the brain for a new decision.
        /// </summary>
        public void RequestDecision()
        {
            requestDecision = true;
            RequestAction();
        }

        /// <summary>
        /// Is called then the agent must perform a new action.
        /// </summary>
        public void RequestAction()
        {
            requestAction = true;
        }

        /// <summary>
        /// Indicates if the agent has reached his maximum number of steps.
        /// </summary>
        /// <returns>
        /// <c>true</c>, if max step reached was reached, <c>false</c> otherwise.
        /// </returns>
        public bool IsMaxStepReached()
        {
            return maxStepReached;
        }

        /// <summary>
        /// Indicates if the agent is done
        /// </summary>
        /// <returns>
        /// <c>true</c>, if the agent is done, <c>false</c> otherwise.
        /// </returns>
        public bool IsDone()
        {
            return done;
        }

        /// Helper function that resets all the data structures associated with
        /// the agent. Typically used when the agent is being initialized or reset
        /// at the end of an episode.
        void ResetData()
        {
            if (brain == null)
            {
                return;
            }

            BrainParameters param = brain.brainParameters;
            actionMasker = new ActionMasker(param);
            if (param.vectorActionSpaceType == SpaceType.continuous)
            {
                action.vectorActions = new float[param.vectorActionSize[0]];
                info.storedVectorActions = new float[param.vectorActionSize[0]];
            }
            else
            {
                action.vectorActions = new float[param.vectorActionSize.Length];
                info.storedVectorActions = new float[param.vectorActionSize.Length];
            }

            if (info.textObservation == null)
                info.textObservation = "";
            action.textActions = "";
            info.memories = new List<float>();
            action.memories = new List<float>();
            info.vectorObservation =
                new List<float>(param.vectorObservationSize);
            info.stackedVectorObservation =
                new List<float>(param.vectorObservationSize
                                * brain.brainParameters.numStackedVectorObservations);
            info.stackedVectorObservation.AddRange(
                new float[param.vectorObservationSize
                          * param.numStackedVectorObservations]);

            info.visualObservations = new List<Texture2D>();
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
        /// Sends the Agent info to the linked Brain.
        /// </summary>
        void SendInfoToBrain()
        {
            if (brain == null)
            {
                return;
            }

            info.memories = action.memories;
            info.storedVectorActions = action.vectorActions;
            info.storedTextActions = action.textActions;
            info.vectorObservation.Clear();
            actionMasker.ResetMask();
            CollectObservations();
            info.actionMasks = actionMasker.GetMask();

            BrainParameters param = brain.brainParameters;
            if (info.vectorObservation.Count != param.vectorObservationSize)
            {
                throw new UnityAgentsException(string.Format(
                    "Vector Observation size mismatch between continuous " +
                    "agent {0} and brain {1}. " +
                    "Was Expecting {2} but received {3}. ",
                    gameObject.name, brain.name,
                    brain.brainParameters.vectorObservationSize,
                    info.vectorObservation.Count));
            }

            info.stackedVectorObservation.RemoveRange(
                0, param.vectorObservationSize);
            info.stackedVectorObservation.AddRange(info.vectorObservation);

            info.visualObservations.Clear();
            if (param.cameraResolutions.Length > agentParameters.agentCameras.Count)
            {
                throw new UnityAgentsException(string.Format(
                    "Not enough cameras for agent {0} : Bain {1} expecting at " +
                    "least {2} cameras but only {3} were present.",
                    gameObject.name, brain.name,
                    brain.brainParameters.cameraResolutions.Length,
                    agentParameters.agentCameras.Count));
            }

            for (int i = 0; i < brain.brainParameters.cameraResolutions.Length; i++)
            {
                ObservationToTexture(
                    agentParameters.agentCameras[i],
                    param.cameraResolutions[i].width,
                    param.cameraResolutions[i].height,
                    ref textureArray[i]);
                info.visualObservations.Add(textureArray[i]);
            }

            info.reward = reward;
            info.done = done;
            info.maxStepReached = maxStepReached;
            info.id = id;

            brain.SendState(this, info);

            if (recorder != null && recorder.record && Application.isEditor)
            {
                recorder.WriteExperience(info);
            }

            info.textObservation = "";
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
        ///     - <see cref="AddVectorObs(float[])"/>
        ///     - <see cref="AddVectorObs(List{float})"/>
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
            actionMasker.SetActionMask(0, actionIndices);
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
            actionMasker.SetActionMask(0, new int[1] {actionIndex});
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
            actionMasker.SetActionMask(branch, new int[1] {actionIndex});
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
            actionMasker.SetActionMask(branch, actionIndices);
        }


        /// <summary>
        /// Adds a float observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 1.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(float observation)
        {
            info.vectorObservation.Add(observation);
        }

        /// <summary>
        /// Adds an integer observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 1.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(int observation)
        {
            info.vectorObservation.Add(observation);
        }

        /// <summary>
        /// Adds an Vector3 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 3.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector3 observation)
        {
            info.vectorObservation.Add(observation.x);
            info.vectorObservation.Add(observation.y);
            info.vectorObservation.Add(observation.z);
        }

        /// <summary>
        /// Adds an Vector2 observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 2.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Vector2 observation)
        {
            info.vectorObservation.Add(observation.x);
            info.vectorObservation.Add(observation.y);
        }

        /// <summary>
        /// Adds a collection of float observations to the vector observations of the agent.
        /// Increases the size of the agents vector observation by size of the collection.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(IEnumerable<float> observation)
        {
            info.vectorObservation.AddRange(observation);
        }

        /// <summary>
        /// Adds a quaternion observation to the vector observations of the agent.
        /// Increases the size of the agents vector observation by 4.
        /// </summary>
        /// <param name="observation">Observation.</param>
        protected void AddVectorObs(Quaternion observation)
        {
            info.vectorObservation.Add(observation.x);
            info.vectorObservation.Add(observation.y);
            info.vectorObservation.Add(observation.z);
            info.vectorObservation.Add(observation.w);
        }

        /// <summary>
        /// Adds a boolean observation to the vector observation of the agent.
        /// Increases the size of the agent's vector observation by 1.
        /// </summary>
        /// <param name="observation"></param>
        protected void AddVectorObs(bool observation)
        {
            info.vectorObservation.Add(observation ? 1f : 0f);
        }

        protected void AddVectorObs(int observation, int range)
        {
            float[] oneHotVector = new float[range];
            oneHotVector[observation] = 1;
            info.vectorObservation.AddRange(oneHotVector);
        }

        /// <summary>
        /// Sets the text observation.
        /// </summary>
        /// <param name="textObservation">The text observation.</param>
        public void SetTextObs(string textObservation)
        {
            info.textObservation = textObservation;
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
        /// An internal reset method that updates internal data structures in
        /// addition to calling <see cref="AgentReset"/>.
        /// </summary>
        void _AgentReset()
        {
            ResetData();
            stepCount = 0;
            AgentReset();
        }

        /// <summary>
        /// Updates the vector action.
        /// </summary>
        /// <param name="vectorActions">Vector actions.</param>
        public void UpdateVectorAction(float[] vectorActions)
        {
            action.vectorActions = vectorActions;
        }

        /// <summary>
        /// Updates the memories action.
        /// </summary>
        /// <param name="memories">Memories.</param>
        public void UpdateMemoriesAction(List<float> memories)
        {
            action.memories = memories;
        }
        
        public void AppendMemoriesAction(List<float> memories)
        {
            action.memories.AddRange(memories);
        }

        /// <summary>
        /// Updates the text action.
        /// </summary>
        /// <param name="textActions">Text actions.</param>
        public void UpdateTextAction(string textActions)
        {
            action.textActions = textActions;
        }

        /// <summary>
        /// Updates the value of the agent.
        /// </summary>
        /// <param name="textActions">Text actions.</param>
        public void UpdateValueAction(float value)
        {
            action.value = value;
        }

        protected float GetValueEstimate()
        {
            return action.value;
        }

        /// <summary>
        /// Scales continous action from [-1, 1] to arbitrary range.
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
        /// Sets the status of the agent.
        /// </summary>
        /// <param name="academyMaxStep">If set to <c>true</c> 
        /// The agent must set maxStepReached.</param>
        /// <param name="academyDone">If set to <c>true</c> 
        /// The agent must set done.</param>
        /// <param name="academyStepCounter">Number of current steps in episode</param>
        void SetStatus(bool academyMaxStep, bool academyDone, int academyStepCounter)
        {
            if (academyDone)
            {
                academyStepCounter = 0;
            }

            MakeRequests(academyStepCounter);
            if (academyMaxStep)
            {
                maxStepReached = true;
            }

            // If the Academy needs to reset, the agent should reset 
            // even if it reseted recently.
            if (academyDone)
            {
                Done();
                hasAlreadyReset = false;
            }
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
                        if (!hasAlreadyReset)
                        {
                            // If event based, the agent can reset as soon
                            // as it is done
                            _AgentReset();
                            hasAlreadyReset = true;
                        }
                    }
                    else if (requestDecision)
                    {
                        // If not event based, the agent must wait to request a
                        // decsion before reseting to keep multiple agents in sync.
                        _AgentReset();
                    }
                }
                else
                {
                    terminate = true;
                    RequestDecision();
                }
            }
        }

        /// <summary>
        /// Signals the agent that it must sent its decision to the brain.
        /// </summary>
        void SendInfo()
        {
            if (requestDecision)
            {
                SendInfoToBrain();
                ResetReward();
                done = false;
                maxStepReached = false;
                requestDecision = false;

                hasAlreadyReset = false;
            }
        }

        /// Used by the brain to make the agent perform a step.
        void AgentStep()
        {
            if (terminate)
            {
                terminate = false;
                ResetReward();
                done = false;
                maxStepReached = false;
                requestDecision = false;
                requestAction = false;

                hasAlreadyReset = false;
                OnDisable();
                AgentOnDone();
            }

            if ((requestAction) && (brain != null))
            {
                requestAction = false;
                AgentAction(action.vectorActions, action.textActions);
            }

            if ((stepCount >= agentParameters.maxStep)
                && (agentParameters.maxStep > 0))
            {
                maxStepReached = true;
                Done();
            }

            stepCount += 1;
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

        /// <summary>
        /// Converts a camera and correspinding resolution to a 2D texture.
        /// </summary>
        /// <returns>The 2D texture.</returns>
        /// <param name="obsCamera">Camera.</param>
        /// <param name="width">Width of resulting 2D texture.</param>
        /// <param name="height">Height of resulting 2D texture.</param>
        /// <param name="texture2D">Texture2D to render to.</param>
        public static void ObservationToTexture(Camera obsCamera, int width, int height, ref Texture2D texture2D)
        {
            Rect oldRec = obsCamera.rect;
            obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
            var depth = 24;
            var format = RenderTextureFormat.Default;
            var readWrite = RenderTextureReadWrite.Default;

            var tempRT =
                RenderTexture.GetTemporary(width, height, depth, format, readWrite);

            if (width != texture2D.width || height != texture2D.height)
            {
                texture2D.Resize(width, height);
            }

            var prevActiveRT = RenderTexture.active;
            var prevCameraRT = obsCamera.targetTexture;

            // render to offscreen texture (readonly from CPU side)
            RenderTexture.active = tempRT;
            obsCamera.targetTexture = tempRT;

            obsCamera.Render();

            texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);
            texture2D.Apply();
            obsCamera.targetTexture = prevCameraRT;
            obsCamera.rect = oldRec;
            RenderTexture.active = prevActiveRT;
            RenderTexture.ReleaseTemporary(tempRT);
        }
    }
}
