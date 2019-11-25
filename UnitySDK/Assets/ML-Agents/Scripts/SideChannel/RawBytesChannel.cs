using System.Collections.Generic;
namespace MLAgents
{
    public class RawBytesChannel : SideChannel
    {

        private List<byte[]> m_MessagesReceived = new List<byte[]>();
        private int m_ChannelId;


        public RawBytesChannel(int channelId = 0)
        {
            m_ChannelId = channelId;
        }
        public override int ChannelType()
        {
            return (int)SideChannelType.RawBytesChannelStart + m_ChannelId;
        }

        public override void OnMessageReceived(byte[] data)
        {
            m_MessagesReceived.Add(data);
        }

        public void SendRawBytes(byte[] data)
        {
            QueueMessageToSend(data);
        }

        public IList<byte[]> ReceiveRawBytes()
        {
            var result = new List<byte[]>();
            result.AddRange(m_MessagesReceived);
            m_MessagesReceived.Clear();
            return result;
        }
    }
}
