using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System;
using System.IO;
using System.Text;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Utility class for reading the data sent to the SideChannel.
    /// </summary>
    public class IncomingMessage : IDisposable
    {
        byte[] m_Data;
        Stream m_Stream;
        BinaryReader m_Reader;

        /// <summary>
        /// Construct an IncomingMessage from the byte array.
        /// </summary>
        /// <param name="data"></param>
        public IncomingMessage(byte[] data)
        {
            m_Data = data;
            m_Stream = new MemoryStream(data);
            m_Reader = new BinaryReader(m_Stream);
        }

        /// <summary>
        /// Read a boolean value from the message.
        /// </summary>
        /// <param name="defaultValue">Default value to use if the end of the message is reached.</param>
        /// <returns></returns>
        public bool ReadBoolean(bool defaultValue = false)
        {
            return CanReadMore() ? m_Reader.ReadBoolean() : defaultValue;
        }

        /// <summary>
        /// Read an integer value from the message.
        /// </summary>
        /// <param name="defaultValue">Default value to use if the end of the message is reached.</param>
        /// <returns></returns>
        public int ReadInt32(int defaultValue = 0)
        {
            return CanReadMore() ? m_Reader.ReadInt32() : defaultValue;
        }

        /// <summary>
        /// Read a float value from the message.
        /// </summary>
        /// <param name="defaultValue">Default value to use if the end of the message is reached.</param>
        /// <returns></returns>
        public float ReadFloat32(float defaultValue = 0.0f)
        {
            return CanReadMore() ? m_Reader.ReadSingle() : defaultValue;
        }

        /// <summary>
        /// Read a string value from the message.
        /// </summary>
        /// <param name="defaultValue">Default value to use if the end of the message is reached.</param>
        /// <returns></returns>
        public string ReadString(string defaultValue = default)
        {
            if (!CanReadMore())
            {
                return defaultValue;
            }

            var strLength = ReadInt32();
            var str = Encoding.ASCII.GetString(m_Reader.ReadBytes(strLength));
            return str;
        }

        /// <summary>
        /// Reads a list of floats from the message. The length of the list is stored in the message.
        /// </summary>
        /// <param name="defaultValue">Default value to use if the end of the message is reached.</param>
        /// <returns></returns>
        public IList<float> ReadFloatList(IList<float> defaultValue = default)
        {
            if (!CanReadMore())
            {
                return defaultValue;
            }

            var len = ReadInt32();
            var output = new float[len];
            for (var i = 0; i < len; i++)
            {
                output[i] = ReadFloat32();
            }

            return output;
        }

        /// <summary>
        /// Gets the original data of the message. Note that this will return all of the data,
        /// even if part of it has already been read.
        /// </summary>
        /// <returns></returns>
        public byte[] GetRawBytes()
        {
            return m_Data;
        }

        /// <summary>
        /// Clean up the internal storage.
        /// </summary>
        public void Dispose()
        {
            m_Reader?.Dispose();
            m_Stream?.Dispose();
        }

        /// <summary>
        /// Whether or not there is more data left in the stream that can be read.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        bool CanReadMore()
        {
            return m_Stream.Position < m_Stream.Length;
        }
    }
}
