using System.Collections.Generic;

namespace MLAgents
{
    public enum SideChannelType
    {
        // Invalid side channel
        Invalid = 0,
        // Reserved for the FloatPropertiesChannel.
        FloatProperties = 1,
        //Reserved for the EngineConfigurationChannel.
        EngineSettings = 2,
        // Raw bytes channels should start here to avoid conflicting with other Unity ones.
        RawBytesChannelStart = 1000,
        // custom side channels should start here to avoid conflicting with Unity ones.
        UserSideChannelStart = 2000,
    }

    public abstract class SideChannel
    {
        // The list of messages (byte arrays) that need to be sent to Python via the communicator.
        // Should only ever be read and cleared by a ICommunicator object.
        public List<byte[]> MessageQueue = new List<byte[]>();

        /// <summary>
        /// An int identifier for the SideChannel. Ensures that there is only ever one side channel
        /// of each type. Ensure the Unity side channels will be linked to their Python equivalent.
        /// </summary>
        /// <returns> The integer identifier of the SideChannel</returns>
        public abstract int ChannelType();

        /// <summary>
        /// Is called by the communicator every time a message is received from Python by the SideChannel.
        /// Can be called multiple times per simulation step if multiple messages were sent.
        /// </summary>
        /// <param name="data"> the payload of the message.</param>
        public abstract void OnMessageReceived(byte[] data);

        /// <summary>
        /// Queues a message to be sent to Python during the next simulation step.
        /// </summary>
        /// <param name="data"> The byte array of data to be sent to Python.</param>
        protected void QueueMessageToSend(byte[] data)
        {
            MessageQueue.Add(data);
        }
    }
}
