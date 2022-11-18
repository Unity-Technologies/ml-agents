using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side channels provide an alternative mechanism of sending/receiving data from Unity
    /// to Python that is outside of the traditional machine learning loop. ML-Agents provides
    /// some specific implementations of side channels, but users can create their own.
    ///
    /// To create your own, you'll need to create two, new mirrored classes, one in Unity (by
    /// extending <see cref="SideChannel"/>) and another in Python by extending a Python class
    /// also called SideChannel. Then, within your project, use
    /// <see cref="SideChannelManager.RegisterSideChannel"/> and
    /// <see cref="SideChannelManager.UnregisterSideChannel"/> to register and unregister your
    /// custom side channel.
    /// </summary>
    public abstract class SideChannel
    {
        // The list of messages (byte arrays) that need to be sent to Python via the communicator.
        // Should only ever be read and cleared by a ICommunicator object.
        internal List<byte[]> MessageQueue = new List<byte[]>();

        /// <summary>
        /// An int identifier for the SideChannel. Ensures that there is only ever one side channel
        /// of each type. Ensure the Unity side channels will be linked to their Python equivalent.
        /// </summary>
        /// <returns> The integer identifier of the SideChannel.</returns>
        public Guid ChannelId
        {
            get;
            protected set;
        }

        internal void ProcessMessage(byte[] msg)
        {
            try
            {
                using (var incomingMsg = new IncomingMessage(msg))
                {
                    OnMessageReceived(incomingMsg);
                }
            }
            catch (Exception ex)
            {
                // Catch all errors in the sidechannel processing, so that a single
                // bad SideChannel implementation doesn't take everything down with it.
                Debug.LogError($"Error processing SideChannel message: {ex}.\nThe message will be skipped.");
            }
        }

        /// <summary>
        /// Is called by the communicator every time a message is received from Python by the SideChannel.
        /// Can be called multiple times per simulation step if multiple messages were sent.
        /// </summary>
        /// <param name="msg">The incoming message.</param>
        protected abstract void OnMessageReceived(IncomingMessage msg);

        /// <summary>
        /// Queues a message to be sent to Python during the next simulation step.
        /// </summary>
        /// <param name="msg"> The byte array of data to be sent to Python.</param>
        protected void QueueMessageToSend(OutgoingMessage msg)
        {
            MessageQueue.Add(msg.ToByteArray());
        }
    }
}
