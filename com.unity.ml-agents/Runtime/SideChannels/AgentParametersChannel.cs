using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.MLAgents.SideChannels
{
    internal class AgentParametersChannel : SideChannel
    {
        Dictionary<int, Dictionary<string, float>> m_Parameters = new Dictionary<int, Dictionary<string, float>>();

        const string k_EnvParamsId = "534c891e-810f-11ea-a9d0-822485860401";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal AgentParametersChannel()
        {
            ChannelId = new Guid(k_EnvParamsId);
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var episodeId = msg.ReadInt32();
            var key = msg.ReadString();
            var value = msg.ReadFloat32();
            if(!m_Parameters.ContainsKey(episodeId))
            {
                m_Parameters[episodeId] = new Dictionary<string, float>();
            }
            m_Parameters[episodeId][key] = value;
        }

        /// <summary>
        /// Returns the parameter value associated with the provided key. Returns the default
        /// value if one doesn't exist.
        /// </summary>
        /// <param name="key">Parameter key.</param>
        /// <param name="defaultValue">Default value to return.</param>
        /// <returns></returns>
        public float GetWithDefault(int episodeId, string key, float defaultValue)
        {
            float value = defaultValue;
            bool hasKey = false;
            Dictionary<string, float> agent_dict;
            if(m_Parameters.TryGetValue(episodeId, out agent_dict))
            {
                agent_dict.TryGetValue(key, out value);
            }
            return value;
        }

        /// <summary>
        /// Returns all parameter keys that have a registered value.
        /// </summary>
        /// <returns></returns>
        public IList<string> ListParameters(int episodeId)
        {
            Dictionary<string, float> agent_dict;
            m_Parameters.TryGetValue(episodeId, out agent_dict);
            return new List<string>(agent_dict.Keys);
        }
    }
}
