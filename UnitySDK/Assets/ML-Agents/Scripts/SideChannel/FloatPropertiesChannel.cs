using System.Collections.Generic;
using System.IO;
using System;
using System.Text;

namespace MLAgents
{

    public interface IFloatProperties
    {
        /// <summary>
        /// Sets one of the float properties of the environment. This data will be sent to Python.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="value"> The float value of the property.</param>
        void SetProperty(string key, float value);

        /// <summary>
        /// Get an Environment property with a default value. If there is a value for this property,
        /// it will be returned, otherwise, the default value will be returned.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="defaultValue"> The default value of the property.</param>
        /// <returns></returns>
        float GetPropertyWithDefault(string key, float defaultValue);

        /// <summary>
        /// Registers an action to be performed everytime the property is changed.
        /// </summary>
        /// <param name="key"> The string identifier of the property.</param>
        /// <param name="action"> The action that ill be performed. Takes a float as input.</param>
        void RegisterCallback(string key, Action<float> action);

        /// <summary>
        /// Returns a list of all the string identifiers of the properties currently present.
        /// </summary>
        /// <returns> The list of string identifiers </returns>
        IList<string> ListProperties();
    }

    public class FloatPropertiesChannel : SideChannel, IFloatProperties
    {

        private Dictionary<string, float> m_FloatProperties = new Dictionary<string, float>();
        private Dictionary<string, Action<float>> m_RegisteredActions = new Dictionary<string, Action<float>>();

        public override int ChannelType()
        {
            return (int)SideChannelType.FloatProperties;
        }

        public override void OnMessageReceived(byte[] data)
        {
            var kv = DeserializeMessage(data);
            m_FloatProperties[kv.Key] = kv.Value;
            if (m_RegisteredActions.ContainsKey(kv.Key))
            {
                m_RegisteredActions[kv.Key].Invoke(kv.Value);
            }
        }

        public void SetProperty(string key, float value)
        {
            m_FloatProperties[key] = value;
            QueueMessageToSend(SerializeMessage(key, value));
            if (m_RegisteredActions.ContainsKey(key))
            {
                m_RegisteredActions[key].Invoke(value);
            }
        }

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

        public void RegisterCallback(string key, Action<float> action)
        {
            m_RegisteredActions[key] = action;
        }

        public IList<string> ListProperties()
        {
            return new List<string>(m_FloatProperties.Keys);
        }

        private static KeyValuePair<string, float> DeserializeMessage(byte[] data)
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

        private static byte[] SerializeMessage(string key, float value)
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
