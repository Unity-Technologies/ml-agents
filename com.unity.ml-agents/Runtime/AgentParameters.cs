using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;

namespace Unity.MLAgents
{
    /// <summary>
    /// A container for the Environment Parameters that may be modified during training.
    /// The keys for those parameters are defined in the trainer configurations and the
    /// the values are generated from the training process in features such as Curriculum Learning
    /// and Environment Parameter Randomization.
    ///
    /// One current assumption for all the environment parameters is that they are of type float.
    /// </summary>
    public sealed class AgentParameters
    {
        /// <summary>
        /// The side channel that is used to receive the new parameter values.
        /// </summary>
        readonly AgentParametersChannel m_Channel;

        /// <summary>
        /// Constructor.
        /// </summary>
        internal AgentParameters()
        {
            m_Channel = new AgentParametersChannel();
            SideChannelManager.RegisterSideChannel(m_Channel);
        }

        /// <summary>
        /// Returns the parameter value for the specified key. Returns the default value provided
        /// if this parameter key does not have a value. Only returns a parameter value if it is
        /// of type float.
        /// </summary>
        /// <param name="key">The parameter key</param>
        /// <param name="defaultValue">Default value for this parameter.</param>
        /// <returns></returns>
        public float GetWithDefault(int episodeId, string key, float defaultValue)
        {
            return m_Channel.GetWithDefault(episodeId, key, defaultValue);
        }


        internal void Dispose()
        {
            SideChannelManager.UnregisterSideChannel(m_Channel);
        }
    }
}
