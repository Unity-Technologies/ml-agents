using System.Collections.Generic;
using System;
using System.IO;
using System.Text;

namespace MLAgents.SideChannels
{
    /// <summary>
    /// Side channels provide an alternative mechanism of sending/receiving data from Unity
    /// to Python that is outside of the traditional machine learning loop. ML-Agents provides
    /// some specific implementations of side channels, but users can create their own.
    /// </summary>
    public abstract class SideChannel
    {
        // The list of messages (byte arrays) that need to be sent to Python via the communicator.
        // Should only ever be read and cleared by a ICommunicator object.
        internal List<byte[]> MessageQueue = new List<byte[]>();

        /// <summary>
        /// An int identifier for the SideChannel. Ensures that there is only ever one side channel
        /// of each type. Ensure the Unity side channels will be linked to their Python equivalent.
        /// </summary>
        /// <returns> The integer identifier of the SideChannel.</returns>
        public Guid ChannelId
        {
            get;
            protected set;
        }

        /// <summary>
        /// Is called by the communicator every time a message is received from Python by the SideChannel.
        /// Can be called multiple times per simulation step if multiple messages were sent.
        /// </summary>
        /// <param name="msg">The incoming message.</param>
        public abstract void OnMessageReceived(IncomingMessage msg);

        /// <summary>
        /// Queues a message to be sent to Python during the next simulation step.
        /// </summary>
        /// <param name="data"> The byte array of data to be sent to Python.</param>
        protected void QueueMessageToSend(OutgoingMessage msg)
        {
            MessageQueue.Add(msg.ToByteArray());
        }
    }

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
        /// Read a boolan value from the message.
        /// </summary>
        /// <returns></returns>
        public bool ReadBoolean()
        {
            return m_Reader.ReadBoolean();
        }

        /// <summary>
        /// Read an integer value from the message.
        /// </summary>
        /// <returns></returns>
        public int ReadInt32()
        {
            return m_Reader.ReadInt32();
        }

        /// <summary>
        /// Read a float value from the message.
        /// </summary>
        /// <returns></returns>
        public float ReadFloat32()
        {
            return m_Reader.ReadSingle();
        }

        /// <summary>
        /// Read a string value from the message.
        /// </summary>
        /// <returns></returns>
        public string ReadString()
        {
            var strLength = ReadInt32();
            var str = Encoding.ASCII.GetString(m_Reader.ReadBytes(strLength));
            return str;
        }

        /// <summary>
        /// Reads a list of floats from the message. The length of the list is stored in the message.
        /// </summary>
        /// <returns></returns>
        public IList<float> ReadFloatList()
        {
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
    }

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
            // Reset first.
            m_Stream.Seek(0, SeekOrigin.Begin);
            m_Stream.SetLength(0);

            // Then append the data
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
