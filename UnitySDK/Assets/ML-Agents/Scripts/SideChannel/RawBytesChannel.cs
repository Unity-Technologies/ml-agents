using System.Collections.Generic;
namespace MLAgents
{
    public class RawBytesChannel : SideChannel
    {

        private List<byte[]> m_MessagesReceived = new List<byte[]>();

        public override int ChannelType() { return 0; }

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
