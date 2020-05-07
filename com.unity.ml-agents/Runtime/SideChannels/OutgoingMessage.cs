using System.Collections.Generic;
using System;
using System.IO;
using System.Text;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Utility class for forming the data that is sent to the SideChannel.
    /// </summary>
    public class OutgoingMessage : IDisposable
    {
        BinaryWriter m_Writer;
        MemoryStream m_Stream;

        /// <summary>
        /// Create a new empty OutgoingMessage.
        /// </summary>
        public OutgoingMessage()
        {
            m_Stream = new MemoryStream();
            m_Writer = new BinaryWriter(m_Stream);
        }

        /// <summary>
        /// Clean up the internal storage.
        /// </summary>
        public void Dispose()
        {
            m_Writer?.Dispose();
            m_Stream?.Dispose();
        }

        /// <summary>
        /// Write a boolean value to the message.
        /// </summary>
        /// <param name="b"></param>
        public void WriteBoolean(bool b)
        {
            m_Writer.Write(b);
        }

        /// <summary>
        /// Write an interger value to the message.
        /// </summary>
        /// <param name="i"></param>
        public void WriteInt32(int i)
        {
            m_Writer.Write(i);
        }

        /// <summary>
        /// Write a float values to the message.
        /// </summary>
        /// <param name="f"></param>
        public void WriteFloat32(float f)
        {
            m_Writer.Write(f);
        }

        /// <summary>
        /// Write a string value to the message.
        /// </summary>
        /// <param name="s"></param>
        public void WriteString(string s)
        {
            var stringEncoded = Encoding.ASCII.GetBytes(s);
            m_Writer.Write(stringEncoded.Length);
            m_Writer.Write(stringEncoded);
        }

        /// <summary>
        /// Write a list or array of floats to the message.
        /// </summary>
        /// <param name="floatList"></param>
        public void WriteFloatList(IList<float> floatList)
        {
            WriteInt32(floatList.Count);
            foreach (var f in floatList)
            {
                WriteFloat32(f);
            }
        }

        /// <summary>
        /// Overwrite the message with a specific byte array.
        /// </summary>
        /// <param name="data"></param>
        public void SetRawBytes(byte[] data)
        {
            // Reset first. Set the length to zero so that if there's more data than we're going to
            // write, we don't have any of the original data.
            m_Stream.Seek(0, SeekOrigin.Begin);
            m_Stream.SetLength(0);

            // Then append the data. Increase the capacity if needed (but don't shrink it).
            m_Stream.Capacity = (m_Stream.Capacity < data.Length) ? data.Length : m_Stream.Capacity;
            m_Stream.Write(data, 0, data.Length);
        }

        /// <summary>
        /// Read the byte array of the message.
        /// </summary>
        /// <returns></returns>
        internal byte[] ToByteArray()
        {
            return m_Stream.ToArray();
        }
    }
}
