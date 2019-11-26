using System.Collections.Generic;
using System.IO;
using System;
using System.Text;

namespace MLAgents
{

    public interface IFloatProperties
    {
        void SetProperty(string key, float value);
        float GetPropertyWithDefault(string key, float defaultValue = 0f);
        void RegisterCallback(string key, Action<float> action);
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
        }

        public float GetPropertyWithDefault(string key, float defaultValue = 0f)
        {
            if (m_FloatProperties.ContainsKey(key))
            {
                return m_FloatProperties[key];
            }
            else
            {
                QueueMessageToSend(SerializeMessage(key, defaultValue));
                m_FloatProperties[key] = defaultValue;
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
