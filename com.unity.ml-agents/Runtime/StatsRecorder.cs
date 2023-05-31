using Unity.MLAgents.SideChannels;

namespace Unity.MLAgents
{
    /// <summary>
    /// Determines the behavior of how multiple stats within the same summary period are combined.
    /// </summary>
    public enum StatAggregationMethod
    {
        /// <summary>
        /// Values within the summary period are averaged before reporting.
        /// </summary>
        Average = 0,

        /// <summary>
        /// Only the most recent value is reported.
        /// To avoid conflicts when training with multiple concurrent environments, only
        /// stats from worker index 0 will be tracked.
        /// </summary>
        MostRecent = 1,

        /// <summary>
        /// Values within the summary period are summed up before reporting.
        /// </summary>
        Sum = 2,

        /// <summary>
        /// Values within the summary period are reported as a histogram.
        /// </summary>
        Histogram = 3
    }

    /// <summary>
    /// Add stats (key-value pairs) for reporting. These values will sent these to a StatsReporter
    /// instance, which means the values will appear in the TensorBoard summary, as well as trainer
    /// gauges. You can nest stats in TensorBoard by adding "/" in the name (e.g. "Agent/Health"
    /// and "Agent/Wallet"). Note that stats are only written to TensorBoard each summary_frequency
    /// steps (a trainer configuration). If a stat is received multiple times, within that period
    /// then the values will be aggregated using the <see cref="StatAggregationMethod"/> provided.
    /// </summary>
    public sealed class StatsRecorder
    {
        /// <summary>
        /// The side channel that is used to receive the new parameter values.
        /// </summary>
        readonly StatsSideChannel m_Channel;

        /// <summary>
        /// Constructor.
        /// </summary>
        internal StatsRecorder()
        {
            m_Channel = new StatsSideChannel();
            SideChannelManager.RegisterSideChannel(m_Channel);
        }

        /// <summary>
        /// Add a stat value for reporting.
        /// </summary>
        /// <param name="key">The stat name.</param>
        /// <param name="value">
        /// The stat value. You can nest stats in TensorBoard by using "/".
        /// </param>
        /// <param name="aggregationMethod">
        /// How multiple values sent in the same summary window should be treated.
        /// </param>
        public void Add(
            string key,
            float value,
            StatAggregationMethod aggregationMethod = StatAggregationMethod.Average)
        {
            m_Channel.AddStat(key, value, aggregationMethod);
        }

        internal void Dispose()
        {
            SideChannelManager.UnregisterSideChannel(m_Channel);
        }
    }
}
