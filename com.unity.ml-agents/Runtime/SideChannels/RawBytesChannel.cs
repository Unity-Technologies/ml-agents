using System.Collections.Generic;
using System;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side channel for managing raw bytes of data. It is up to the clients of this side channel
    /// to interpret the messages.
    /// </summary>
    public class RawBytesChannel : SideChannel
    {
        List<byte[]> m_MessagesReceived = new List<byte[]>();

        /// <summary>
        /// RawBytesChannel provides a way to exchange raw byte arrays between Unity and Python.
        /// </summary>
        /// <param name="channelId"> The identifier for the RawBytesChannel. Must be
        /// the same on Python and Unity.</param>
        public RawBytesChannel(Guid channelId)
        {
            ChannelId = channelId;
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            m_MessagesReceived.Add(msg.GetRawBytes());
        }

        /// <summary>
        /// Sends the byte array message to the Python side channel. The message will be sent
        /// alongside the simulation step.
        /// </summary>
        /// <param name="data"> The byte array of data to send to Python.</param>
        public void SendRawBytes(byte[] data)
        {
            using (var msg = new OutgoingMessage())
            {
                msg.SetRawBytes(data);
                QueueMessageToSend(msg);
            }
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
