using System.Collections.Generic;
namespace MLAgents
{
    public class RawBytesChannel : SideChannel
    {
        List<byte[]> m_MessagesReceived = new List<byte[]>();
        int m_ChannelId;

        /// <summary>
        /// RawBytesChannel provides a way to exchange raw byte arrays between Unity and Python.
        /// </summary>
        /// <param name="channelId"> The identifier for the RawBytesChannel. Must be
        /// the same on Python and Unity.</param>
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

        /// <summary>
        /// Sends the byte array message to the Python side channel. The message will be sent
        /// alongside the simulation step.
        /// </summary>
        /// <param name="data"> The byte array of data to send to Python.</param>
        public void SendRawBytes(byte[] data)
        {
            QueueMessageToSend(data);
        }

        /// <summary>
        /// Gets the messages that were sent by python since the last call to
        /// GetAndClearReceivedMessages.
        /// </summary>
        /// <returns> a list of byte array messages that Python has sent.</returns>
        public IList<byte[]> GetAndClearReceivedMessages()
        {
            var result = new List<byte[]>();
            result.AddRange(m_MessagesReceived);
            m_MessagesReceived.Clear();
            return result;
        }

        /// <summary>
        /// Gets the messages that were sent by python since the last call to
        /// GetAndClearReceivedMessages. Note that the messages received will not
        /// be cleared with a call to GetReceivedMessages.
        /// </summary>
        /// <returns> a list of byte array messages that Python has sent.</returns>
        public IList<byte[]> GetReceivedMessages()
        {
            var result = new List<byte[]>();
            result.AddRange(m_MessagesReceived);
            return result;
        }
    }
}
