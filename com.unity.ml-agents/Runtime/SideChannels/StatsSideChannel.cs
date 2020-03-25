using System;
namespace MLAgents.SideChannels
{
    /// <summary>
    /// Determines the behavior of how multiple stats within the same summary period are combined.
    /// </summary>
    public enum StatAggregationMethod
    {
        /// <summary>
        /// Values within the summary period are averaged before reporting.
        /// Note that values from the same C# environment in the same step may replace each other.
        /// </summary>
        Average = 0,

        /// <summary>
        /// Only the most recent value is reported.
        /// To avoid conflicts between multiple environments, the ML Agents environment will only
        /// keep stats from worker index 0.
        /// </summary>
        MostRecent = 1
    }

    /// <summary>
    /// Add stats (key-value pairs) for reporting. The ML Agents environment will send these to a StatsReporter
    /// instance, which means the values will appear in the Tensorboard summary, as well as trainer gauges.
    /// Note that stats are only written every summary_frequency steps; See <see cref="StatAggregationMethod"/>
    /// for options on how multiple values are handled.
    /// </summary>
    public class StatsSideChannel : SideChannel
    {
        const string k_StatsSideChannelDefaultId = "a1d8f7b7-cec8-50f9-b78b-d3e165a78520";

        /// <summary>
        /// Initializes the side channel with the provided channel ID.
        /// The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
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
        /// <param name="key">The stat name.</param>
        /// <param name="value">The stat value. You can nest stats in Tensorboard by using "/". </param>
        /// <param name="aggregationMethod">How multiple values should be treated.</param>
        public void AddStat(
            string key, float value, StatAggregationMethod aggregationMethod = StatAggregationMethod.Average
            )
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
        public override void OnMessageReceived(IncomingMessage msg)
        {
            throw new UnityAgentsException("StatsSideChannel should never receive messages.");
        }
    }
}
