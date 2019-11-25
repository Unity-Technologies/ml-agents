using System.Collections.Generic;

namespace MLAgents
{
    public enum SideChannelType
    {
        FloatProperties = 1,
        EngineSettings = 2,
        // Raw bytes channels should start here to avoid conflicting with other Unity ones.
        RawBytesChannelStart = 1000,
        // custom side channels should start here to avoid conflicting with Unity ones.
        UserSideChannelStart = 2000,
    }

    public abstract class SideChannel
    {

        public List<byte[]> MessageQueue = new List<byte[]>(); // List
        public abstract int ChannelType();

        public abstract void OnMessageReceived(byte[] data);

        protected void QueueMessageToSend(byte[] data)
        {
            MessageQueue.Add(data);
        }

    }
}
