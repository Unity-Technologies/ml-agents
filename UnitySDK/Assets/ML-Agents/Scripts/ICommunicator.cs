using UnityEngine;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{
    public struct CommunicatorParameters
    {
        public int port;
    }

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
    ......UnityRLIntput
    ......UnityRLInitializationIntput

    UnityOutput and UnityInput can be extended to provide functionalities beyond RL
    UnityRLOutput and UnityRLInput can be extended to provide new RL functionalities
     */
    public interface ICommunicator
    {
        /// <summary>
        /// Initialize the communicator by sending the first UnityOutput and receiving the
        /// first UnityInput. The second UnityInput is stored in the unityInput argument.
        /// </summary>
        /// <returns>The first Unity Input.</returns>
        /// <param name="unityOutput">The first Unity Output.</param>
        /// <param name="unityInput">The second Unity input.</param>
        UnityInput Initialize(UnityOutput unityOutput,
            out UnityInput unityInput);

        /// <summary>
        /// Send a UnityOutput and receives a UnityInput.
        /// </summary>
        /// <returns>The next UnityInput.</returns>
        /// <param name="unityOutput">The UnityOutput to be sent.</param>
        UnityInput Exchange(UnityOutput unityOutput);

        /// <summary>
        /// Close the communicator gracefully on both sides of the communication.
        /// </summary>
        void Close();
    }
}
