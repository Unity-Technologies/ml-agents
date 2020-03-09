using System.Collections.Generic;
using System.IO;
using System;
using System.Text;

namespace MLAgents.SideChannels
{
    /// <summary>
    /// Side channel that is comprised of a collection of float variables, represented by
    /// <see cref="IFloatProperties"/>
    /// </summary>
    public class FloatPropertiesChannel : SideChannel
    {
        Dictionary<string, float> m_FloatProperties = new Dictionary<string, float>();
        Dictionary<string, Action<float>> m_RegisteredActions = new Dictionary<string, Action<float>>();
        private const string k_FloatPropertiesDefaultId = "60ccf7d0-4f7e-11ea-b238-784f4387d1f7";

        /// <summary>
        /// Initializes the side channel with the provided channel ID.
        /// </summary>
        /// <param name="channelId">ID for the side channel.</param>
        public FloatPropertiesChannel(Guid channelId = default(Guid))
        {
            if (channelId == default(Guid))
            {
                ChannelId = new Guid(k_FloatPropertiesDefaultId);
            }
            else
            {
                ChannelId = channelId;
            }
        }

        /// <inheritdoc/>
        public override void OnMessageReceived(IncomingMessage msg)
        {
            var key = msg.ReadString();
            var value = msg.ReadFloat32();

            m_FloatProperties[key] = value;
            if (m_RegisteredActions.ContainsKey(key))
            {
                m_RegisteredActions[key].Invoke(value);
            }
        }

        /// <inheritdoc/>
        public void SetProperty(string key, float value)
        {
            m_FloatProperties[key] = value;
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(key);
                msgOut.WriteFloat32(value);
                QueueMessageToSend(msgOut);
            }

            if (m_RegisteredActions.ContainsKey(key))
            {
                m_RegisteredActions[key].Invoke(value);
            }
        }

        /// <inheritdoc/>
        public float GetPropertyWithDefault(string key, float defaultValue)
        {
            if (m_FloatProperties.ContainsKey(key))
            {
                return m_FloatProperties[key];
            }
            else
            {
                return defaultValue;
            }
        }

        /// <inheritdoc/>
        public void RegisterCallback(string key, Action<float> action)
        {
            m_RegisteredActions[key] = action;
        }

        /// <inheritdoc/>
        public IList<string> ListProperties()
        {
            return new List<string>(m_FloatProperties.Keys);
        }
    }
}
