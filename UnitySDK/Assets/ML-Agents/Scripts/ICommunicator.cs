using System;
using System.Collections.Generic;
using UnityEngine;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{
    public struct EnvironmentResetParameters
    {
        /// <summary>
        /// Mapping of string : float which defines which parameters can be
        /// reset from python.
        /// </summary>
        public ResetParameters resetParameters;

        /// <summary>
        /// The protobuf for custom reset parameters.
        /// NOTE: This is the last remaining relic of gRPC protocol
        /// that is left in our code.  We need to decide how to handle this
        /// moving forward.
        /// </summary>
        public CustomResetParametersProto customResetParameters;
    }
    public struct CommunicatorInitParameters
    {
        /// <summary>
        /// Port to listen for connections on.
        /// </summary>
        public int port;
        /// <summary>
        /// The name of the environment.
        /// </summary>
        public string name;
        /// <summary>
        /// The version of the Unity SDK.
        /// </summary>
        public string version;
        /// <summary>
        /// The set of environment parameters defined by the user that will be sent to the communicator.
        /// </summary>
        public EnvironmentResetParameters environmentResetParameters;
    }
    public struct UnityRLInitParameters
    {
        /// <summary>
        /// An RNG seed sent from the python process to Unity.
        /// </summary>
        public int seed;
    }
    public struct UnityRLInputParameters
    {
        /// <summary>
        /// Boolean sent back from python to indicate whether or not training is happening.
        /// </summary>
        public bool isTraining;
    }

    /// <summary>
    /// Delegate for handling quite events sent back from the communicator.
    /// </summary>
    public delegate void QuitCommandHandler();

    /// <summary>
    /// Delegate for handling reset parameter updates sent from the communicator.
    /// </summary>
    /// <param name="resetParams"></param>
    public delegate void ResetCommandHandler(EnvironmentResetParameters resetParams);

    /// <summary>
    /// Delegate to handle UnityRLInputParameters updates from the communicator.
    /// </summary>
    /// <param name="inputParams"></param>
    public delegate void RLInputReceivedHandler(UnityRLInputParameters inputParams);

    /**
    This is the interface of the Communicators.
    This does not need to be modified nor implemented to create a Unity environment.

    When the Unity Communicator is initialized, it will wait for the External Communicator
    to be initialized as well. The two communicators will then exchange their first messages
    that will usually contain information for initialization (information that does not need
    to be resent at each new exchange).

    By convention a Unity input is from External to Unity and a Unity output is from Unity to
    External. Inputs and outputs are relative to Unity.

    By convention, when the Unity Communicator and External Communicator call exchange, the
    exchange is NOT simultaneous but sequential. This means that when a side of the
    communication calls exchange, the other will receive the result of its previous
    exchange call.
    This is what happens when A calls exchange a single time:
    A sends data_1 to B -> B receives data_1 -> B generates and sends data_2 -> A receives data_2
    When A calls exchange, it sends data_1 and receives data_2

    Since the messages are sent back and forth with exchange and simultaneously when calling
    initialize, External sends two messages at initialization.

    The structure of the messages is as follows:
    UnityMessage
    ...Header
    ...UnityOutput
    ......UnityRLOutput
    ......UnityRLInitializationOutput
    ...UnityInput
    ......UnityRLInput
    ......UnityRLInitializationInput

    UnityOutput and UnityInput can be extended to provide functionalities beyond RL
    UnityRLOutput and UnityRLInput can be extended to provide new RL functionalities
     */
    public interface ICommunicator : IBatchedDecisionMaker
    {
        /// <summary>
        /// Quit was received by the communicator.
        /// </summary>
        event QuitCommandHandler QuitCommandReceived;


        /// <summary>
        /// Reset command sent back from the communicator.
        /// </summary>
        event ResetCommandHandler ResetCommandReceived;

        /// <summary>
        /// Unity RL Input was received by the communicator.
        /// </summary>
        event RLInputReceivedHandler RLInputReceived;

        /// <summary>
        /// Sends the academy parameters through the Communicator.
        /// Is used by the academy to send the AcademyParameters to the communicator.
        /// </summary>
        /// <returns>The External Initialization Parameters received.</returns>
        /// <param name="initParameters">The Unity Initialization Parameters to be sent.</param>
        UnityRLInitParameters Initialize(CommunicatorInitParameters initParameters);

        /// <summary>
        /// Registers a new Brain to the Communicator.
        /// </summary>
        /// <param name="name">The name or key uniquely identifying the Brain</param>
        /// <param name="brainParameters">The Parameters for the Brain being registered</param>
        void SubscribeBrain(string name, BrainParameters brainParameters);

        /// <summary>
        /// Gets the AgentActions based on the batching key.
        /// </summary>
        /// <param name="key">A key to identify which actions to get</param>
        /// <returns></returns>
        Dictionary<Agent, AgentAction> GetActions(string key);
    }

    public interface IBatchedDecisionMaker : IDisposable
    {
        void PutObservations(string key, Agent agent);
        void DecideBatch();
    }
}
