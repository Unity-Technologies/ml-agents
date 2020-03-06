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
            else{
                ChannelId = channelId;
            }
        }

        /// <inheritdoc/>
        public override void OnMessageReceived(byte[] data)
        {
            var kv = DeserializeMessage(data);
            m_FloatProperties[kv.Key] = kv.Value;
            if (m_RegisteredActions.ContainsKey(kv.Key))
            {
                m_RegisteredActions[kv.Key].Invoke(kv.Value);
            }
        }

        /// <inheritdoc/>
        public void SetProperty(string key, float value)
        {
            m_FloatProperties[key] = value;
            QueueMessageToSend(SerializeMessage(key, value));
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

        static KeyValuePair<string, float> DeserializeMessage(byte[] data)
        {
            using (var memStream = new MemoryStream(data))
            {
                using (var binaryReader = new BinaryReader(memStream))
                {
                    var keyLength = binaryReader.ReadInt32();
                    var key = Encoding.ASCII.GetString(binaryReader.ReadBytes(keyLength));
                    var value = binaryReader.ReadSingle();
                    return new KeyValuePair<string, float>(key, value);
                }
            }
        }

        static byte[] SerializeMessage(string key, float value)
        {
            using (var memStream = new MemoryStream())
            {
                using (var binaryWriter = new BinaryWriter(memStream))
                {
                    var stringEncoded = Encoding.ASCII.GetBytes(key);
                    binaryWriter.Write(stringEncoded.Length);
                    binaryWriter.Write(stringEncoded);
                    binaryWriter.Write(value);
                    return memStream.ToArray();
                }
            }
        }
    }
}
