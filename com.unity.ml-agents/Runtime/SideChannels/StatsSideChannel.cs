using System;
namespace MLAgents.SideChannels
{
    public class StatsSideChannel : SideChannel
    {
        const string k_StatsSideChannelDefaultId = "a1d8f7b7-cec8-50f9-b78b-d3e165a78520";

        /// <summary>
        /// Initializes the side channel with the provided channel ID.
        /// </summary>
        internal StatsSideChannel()
        {
            ChannelId = new Guid(k_StatsSideChannelDefaultId);
        }

        /// <summary>
        /// Add a stat value for reporting. This will appear in the Tensorboard summary and trainer gauges.
        /// You can nest stats in Tensorboard with "/".
        /// Note that stats are only written to Tensorboard each summary_frequency steps; if a stat is
        /// received multiple times, only the most recent version is used.
        /// To avoid conflicts between multiple environments, only stats from worker index 0 are used.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        public void AddStat(string key, float value)
        {
            using (var msg = new OutgoingMessage())
            {
                msg.WriteString(key);
                msg.WriteFloat32(value);
                QueueMessageToSend(msg);
            }
        }

        /// <inheritdoc/>
        public override void OnMessageReceived(IncomingMessage msg)
        {
            throw new UnityAgentsException("StatsSideChannel should never receive messages.");
        }
    }
}
