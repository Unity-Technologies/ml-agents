using MLAgents.SideChannels;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A container for the Engine Settings that can be specified at the start of
    /// training. These Engine settings are set once at the start of training based on the
    /// training configuration. The actual Engine settings applied may be slightly different
    /// (e.g. TimeScale is clamped). This class enables the retrieval of the final settings. Note
    /// that if the Engine settings are directly changed anywhere in your Project, then the values
    /// returned here may not be reflective of the actual Engine settings.
    /// </summary>
    internal sealed class EngineParameters
    {
        /// <summary>
        /// The side channel that is used to receive the engine configurations.
        /// </summary>
        readonly EngineConfigurationChannel m_Channel;

        /// <summary>
        /// Constructor.
        /// </summary>
        public EngineParameters()
        {
            m_Channel = new EngineConfigurationChannel();
            SideChannelsManager.RegisterSideChannel(m_Channel);
        }

        internal void Dispose()
        {
            SideChannelsManager.UnregisterSideChannel(m_Channel);
        }
    }
}
