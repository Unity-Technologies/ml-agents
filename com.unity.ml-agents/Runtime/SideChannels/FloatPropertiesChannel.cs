using System.Collections.Generic;
using System;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side channel that is comprised of a collection of float variables.
    /// </summary>
    public class FloatPropertiesChannel : SideChannel
    {
        Dictionary<string, float> m_FloatProperties = new Dictionary<string, float>();
        Dictionary<string, Action<float>> m_RegisteredActions = new Dictionary<string, Action<float>>();
        const string k_FloatPropertiesDefaultId = "60ccf7d0-4f7e-11ea-b238-784f4387d1f7";

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
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var key = msg.ReadString();
            var value = msg.ReadFloat32();

            m_FloatProperties[key] = value;

            Action<float> action;
            m_RegisteredActions.TryGetValue(key, out action);
            action?.Invoke(value);
        }

        /// <summary>
        /// Sets one of the float properties of the environment. This data will be sent to Python.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="value"> The float value of the property.</param>
        public void Set(string key, float value)
        {
            m_FloatProperties[key] = value;
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(key);
                msgOut.WriteFloat32(value);
                QueueMessageToSend(msgOut);
            }

            Action<float> action;
            m_RegisteredActions.TryGetValue(key, out action);
            action?.Invoke(value);
        }

        /// <summary>
        /// Get an Environment property with a default value. If there is a value for this property,
        /// it will be returned, otherwise, the default value will be returned.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="defaultValue"> The default value of the property.</param>
        /// <returns></returns>
        public float GetWithDefault(string key, float defaultValue)
        {
            float valueOut;
            bool hasKey = m_FloatProperties.TryGetValue(key, out valueOut);
            return hasKey ? valueOut : defaultValue;
        }

        /// <summary>
        /// Registers an action to be performed everytime the property is changed.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="action"> The action that ill be performed. Takes a float as input.</param>
        public void RegisterCallback(string key, Action<float> action)
        {
            m_RegisteredActions[key] = action;
        }

        /// <summary>
        /// Returns a list of all the string identifiers of the properties currently present.
        /// </summary>
        /// <returns> The list of string identifiers </returns>
        public IList<string> Keys()
        {
            return new List<string>(m_FloatProperties.Keys);
        }
    }
}
