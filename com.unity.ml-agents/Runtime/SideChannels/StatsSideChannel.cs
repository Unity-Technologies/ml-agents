using System;
namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// A Side Channel for sending <see cref="StatsRecorder"/> data.
    /// </summary>
    internal class StatsSideChannel : SideChannel
    {
        const string k_StatsSideChannelDefaultId = "a1d8f7b7-cec8-50f9-b78b-d3e165a78520";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time.
        /// </summary>
        internal StatsSideChannel()
        {
            ChannelId = new Guid(k_StatsSideChannelDefaultId);
        }

        /// <summary>
        /// Add a stat value for reporting.
        /// </summary>
        /// <param name="key">The stat name.</param>
        /// <param name="value">The stat value.</param>
        /// <param name="aggregationMethod">How multiple values should be treated.</param>
        public void AddStat(string key, float value, StatAggregationMethod aggregationMethod)
        {
            using (var msg = new OutgoingMessage())
            {
                msg.WriteString(key);
                msg.WriteFloat32(value);
                msg.WriteInt32((int)aggregationMethod);
                QueueMessageToSend(msg);
            }
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            throw new UnityAgentsException("StatsSideChannel should never receive messages.");
        }
    }
}
