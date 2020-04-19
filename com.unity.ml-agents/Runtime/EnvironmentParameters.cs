using System;
using System.Collections.Generic;
using MLAgents.SideChannels;

namespace MLAgents
{
    /// <summary>
    /// A singleton container for the Environment Parameters that may be modified during training.
    /// The keys for those parameters are defined in the trainer configurations and the
    /// the values are generated from the training process in features such as Curriculum Learning
    /// and Environment Parameter Randomization.
    ///
    /// One current assumption for all the environment parameters is that they are of type float.
    /// </summary>
    public sealed class EnvironmentParameters
    {
        /// <summary>
        /// The side channel that is used to receive the new parameter values.
        /// </summary>
        readonly EnvironmentParametersChannel m_Channel;

        /// <summary>
        /// The singleton instance for this class.
        /// </summary>
        private static EnvironmentParameters s_Instance;

        /// <summary>
        /// Constructor, kept private to make this class a singleton.
        /// </summary>
        private EnvironmentParameters()
        {
            m_Channel = new EnvironmentParametersChannel();
            SideChannelUtils.RegisterSideChannel(m_Channel);
        }

        /// <summary>
        /// Internal Getter for the singleton instance. Initializes if not already initialized.
        /// This method is kept internal to ensure that the initialization is managed by the
        /// Academy. Projects that import the ML-Agents SDK can retrieve the instance via
        /// <see cref="Academy.EnvironmentParameters"/>.
        /// </summary>
        internal static EnvironmentParameters Instance
        {
            get { return s_Instance ?? (s_Instance = new EnvironmentParameters()); }
        }

        /// <summary>
        /// Returns the parameter value for the specified key. Returns the default value provided
        /// if this parameter key does not have a value. Only returns a parameter value if it is
        /// of type float.
        /// </summary>
        /// <param name="key">The parameter key</param>
        /// <param name="defaultValue">Default value for this parameter.</param>
        /// <returns></returns>
        public float GetParameterWithDefault(string key, float defaultValue)
        {
            return m_Channel.GetParameterWithDefault(key, defaultValue);
        }

        /// <summary>
        /// Registers a callback action for the provided parameter key. Will overwrite any
        /// existing action for that parameter. The callback will be called whenever the parameter
        /// receives a value from the training process.
        /// </summary>
        /// <param name="key">The parameter key</param>
        /// <param name="action">The callback action</param>
        public void RegisterCallback(string key, Action<float> action)
        {
            m_Channel.RegisterCallback(key, action);
        }

        /// <summary>
        /// Returns a list of all the parameter keys that have received values.
        /// </summary>
        /// <returns>List of parameter keys.</returns>
        public IList<string> ListParameters()
        {
            return m_Channel.ListParameters();
        }

        internal void Dispose()
        {
            SideChannelUtils.UnregisterSideChannel(m_Channel);
            s_Instance = null;
        }
    }
}
