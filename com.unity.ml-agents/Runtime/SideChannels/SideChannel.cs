using System.Collections.Generic;
using System;
using System.IO;
using System.Text;

namespace MLAgents.SideChannels
{
    /// <summary>
    /// Side channels provide an alternative mechanism of sending/receiving data from Unity
    /// to Python that is outside of the traditional machine learning loop. ML-Agents provides
    /// some specific implementations of side channels, but users can create their own.
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

        /// <summary>
        /// Is called by the communicator every time a message is received from Python by the SideChannel.
        /// Can be called multiple times per simulation step if multiple messages were sent.
        /// </summary>
        /// <param name="msg">The incoming message.</param>
        public abstract void OnMessageReceived(IncomingMessage msg);

        /// <summary>
        /// Queues a message to be sent to Python during the next simulation step.
        /// </summary>
        /// <param name="data"> The byte array of data to be sent to Python.</param>
        protected void QueueMessageToSend(OutgoingMessage msg)
        {
            MessageQueue.Add(msg.ToByteArray());
        }
    }
}
