using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Lists the different data types supported.
    /// </summary>
    internal enum EnvironmentDataTypes
    {
        Float = 0,
        Sampler = 1
    }

    /// <summary>
    /// The types of distributions from which to sample reset parameters.
    /// </summary>
    internal enum SamplerType
    {
        /// <summary>
        /// Samples a reset parameter from a uniform distribution.
        /// </summary>
        Uniform = 0,

        /// <summary>
        /// Samples a reset parameter from a Gaussian distribution.
        /// </summary>
        Gaussian = 1,

        /// <summary>
        /// Samples a reset parameter from a MultiRangeUniform distribution.
        /// </summary>
        MultiRangeUniform = 2
    }

    /// <summary>
    /// A side channel that manages the environment parameter values from Python. Currently
    /// limited to parameters of type float.
    /// </summary>
    internal class EnvironmentParametersChannel : SideChannel
    {
        Dictionary<string, Func<float>> m_Parameters = new Dictionary<string, Func<float>>();
        Dictionary<string, Action<float>> m_RegisteredActions =
            new Dictionary<string, Action<float>>();

        const string k_EnvParamsId = "534c891e-810f-11ea-a9d0-822485860400";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal EnvironmentParametersChannel()
        {
            ChannelId = new Guid(k_EnvParamsId);
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var key = msg.ReadString();
            var type = msg.ReadInt32();
            if ((int)EnvironmentDataTypes.Float == type)
            {
                var value = msg.ReadFloat32();

                m_Parameters[key] = () => value;

                Action<float> action;
                m_RegisteredActions.TryGetValue(key, out action);
                action?.Invoke(value);
            }
            else if ((int)EnvironmentDataTypes.Sampler == type)
            {
                int seed = msg.ReadInt32();
                int samplerType = msg.ReadInt32();
                Func<float> sampler = () => 0.0f;
                if ((int)SamplerType.Uniform == samplerType)
                {
                    float min = msg.ReadFloat32();
                    float max = msg.ReadFloat32();
                    sampler = SamplerFactory.CreateUniformSampler(min, max, seed);
                }
                else if ((int)SamplerType.Gaussian == samplerType)
                {
                    float mean = msg.ReadFloat32();
                    float stddev = msg.ReadFloat32();

                    sampler = SamplerFactory.CreateGaussianSampler(mean, stddev, seed);
                }
                else if ((int)SamplerType.MultiRangeUniform == samplerType)
                {
                    IList<float> intervals = msg.ReadFloatList();
                    sampler = SamplerFactory.CreateMultiRangeUniformSampler(intervals, seed);
                }
                else
                {
                    Debug.LogWarning("EnvironmentParametersChannel received an unknown data type.");
                }
                m_Parameters[key] = sampler;
            }
            else
            {
                Debug.LogWarning("EnvironmentParametersChannel received an unknown data type.");
            }
        }

        /// <summary>
        /// Returns the parameter value associated with the provided key. Returns the default
        /// value if one doesn't exist.
        /// </summary>
        /// <param name="key">Parameter key.</param>
        /// <param name="defaultValue">Default value to return.</param>
        /// <returns></returns>
        public float GetWithDefault(string key, float defaultValue)
        {
            Func<float> valueOut;
            bool hasKey = m_Parameters.TryGetValue(key, out valueOut);
            return hasKey ? valueOut.Invoke() : defaultValue;
        }

        /// <summary>
        /// Registers a callback for the associated parameter key. Will overwrite any existing
        /// actions for this parameter key.
        /// </summary>
        /// <param name="key">The parameter key.</param>
        /// <param name="action">The callback.</param>
        public void RegisterCallback(string key, Action<float> action)
        {
            m_RegisteredActions[key] = action;
        }

        /// <summary>
        /// Returns all parameter keys that have a registered value.
        /// </summary>
        /// <returns></returns>
        public IList<string> ListParameters()
        {
            return new List<string>(m_Parameters.Keys);
        }
    }
}
