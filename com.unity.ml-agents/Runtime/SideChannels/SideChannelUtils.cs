using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace MLAgents.SideChannels
{
    public static class SideChannelUtils
    {

        private static Dictionary<Guid, SideChannel> RegisteredChannels = new Dictionary<Guid, SideChannel>();

        private struct CachedSideChannelMessage
        {
            public Guid ChannelId;
            public byte[] Message;
        }

        private static Queue<CachedSideChannelMessage> m_CachedMessages = new Queue<CachedSideChannelMessage>();

        /// <summary>
        /// Registers a side channel to the communicator. The side channel will exchange
        /// messages with its Python equivalent.
        /// </summary>
        /// <param name="sideChannel"> The side channel to be registered.</param>
        public static void RegisterSideChannel(SideChannel sideChannel)
        {
            var channelId = sideChannel.ChannelId;
            if (RegisteredChannels.ContainsKey(channelId))
            {
                throw new UnityAgentsException(string.Format(
                    "A side channel with type index {0} is already registered. You cannot register multiple " +
                    "side channels of the same id.", channelId));
            }

            // Process any messages that we've already received for this channel ID.
            var numMessages = m_CachedMessages.Count;
            for (int i = 0; i < numMessages; i++)
            {
                var cachedMessage = m_CachedMessages.Dequeue();
                if (channelId == cachedMessage.ChannelId)
                {
                    using (var incomingMsg = new IncomingMessage(cachedMessage.Message))
                    {
                        sideChannel.OnMessageReceived(incomingMsg);
                    }
                }
                else
                {
                    m_CachedMessages.Enqueue(cachedMessage);
                }
            }
            RegisteredChannels.Add(channelId, sideChannel);
        }

        /// <summary>
        /// Unregisters a side channel from the communicator.
        /// </summary>
        /// <param name="sideChannel"> The side channel to be unregistered.</param>
        public static void UnregisterSideChannel(SideChannel sideChannel)
        {
            if (RegisteredChannels.ContainsKey(sideChannel.ChannelId))
            {
                RegisteredChannels.Remove(sideChannel.ChannelId);
            }
        }

        /// <summary>
        /// Unregisters all the side channels from the communicator.
        /// </summary>
        public static void UnregisterAllSideChannels()
        {
            RegisteredChannels = new Dictionary<Guid, SideChannel>();
        }

        /// <summary>
        /// Returns the SideChannel of Type T if there is one registered, or null if it doesn't.
        /// If there are multiple SideChannels of the same type registered, the returned instance is arbitrary.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static T GetSideChannel<T>() where T: SideChannel
        {
            foreach (var sc in RegisteredChannels.Values)
            {
                if (sc.GetType() == typeof(T))
                {
                    return (T) sc;
                }
            }
            return null;
        }

        /// <summary>
        /// Returns all SideChannels of Type T that are registered. Use <see cref="GetSideChannel{T}()"/> if possible,
        /// as that does not make any memory allocations.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static List<T> GetSideChannels<T>() where T: SideChannel
        {
            var output = new List<T>();

            foreach (var sc in RegisteredChannels.Values)
            {
                if (sc.GetType() == typeof(T))
                {
                    output.Add((T) sc);
                }
            }
            return output;
        }

        /// <summary>
        /// Grabs the messages that the registered side channels will send to Python at the current step
        /// into a singe byte array.
        /// </summary>
        /// <returns></returns>
        internal static byte[] GetSideChannelMessage()
        {
            return GetSideChannelMessage(RegisteredChannels);
        }

        /// <summary>
        /// Grabs the messages that the registered side channels will send to Python at the current step
        /// into a singe byte array.
        /// </summary>
        /// <param name="sideChannels"> A dictionary of channel type to channel.</param>
        /// <returns></returns>
        internal static byte[] GetSideChannelMessage(Dictionary<Guid, SideChannel> sideChannels)
        {
            using (var memStream = new MemoryStream())
            {
                using (var binaryWriter = new BinaryWriter(memStream))
                {
                    foreach (var sideChannel in sideChannels.Values)
                    {
                        var messageList = sideChannel.MessageQueue;
                        foreach (var message in messageList)
                        {
                            binaryWriter.Write(sideChannel.ChannelId.ToByteArray());
                            binaryWriter.Write(message.Length);
                            binaryWriter.Write(message);
                        }
                        sideChannel.MessageQueue.Clear();
                    }
                    return memStream.ToArray();
                }
            }
        }

        /// <summary>
        /// Separates the data received from Python into individual messages for each registered side channel.
        /// </summary>
        /// <param name="dataReceived">The byte array of data received from Python.</param>
        internal static void ProcessSideChannelData(byte[] dataReceived)
        {
            ProcessSideChannelData(RegisteredChannels, dataReceived);
        }

        /// <summary>
        /// Separates the data received from Python into individual messages for each registered side channel.
        /// </summary>
        /// <param name="sideChannels">A dictionary of channel type to channel.</param>
        /// <param name="dataReceived">The byte array of data received from Python.</param>
        internal static void ProcessSideChannelData(Dictionary<Guid, SideChannel> sideChannels, byte[] dataReceived)
        {
            while (m_CachedMessages.Count != 0)
            {
                var cachedMessage = m_CachedMessages.Dequeue();
                if (sideChannels.ContainsKey(cachedMessage.ChannelId))
                {
                    using (var incomingMsg = new IncomingMessage(cachedMessage.Message))
                    {
                        sideChannels[cachedMessage.ChannelId].OnMessageReceived(incomingMsg);
                    }
                }
                else
                {
                    Debug.Log(string.Format(
                        "Unknown side channel data received. Channel Id is "
                        + ": {0}", cachedMessage.ChannelId));
                }
            }

            if (dataReceived.Length == 0)
            {
                return;
            }
            using (var memStream = new MemoryStream(dataReceived))
            {
                using (var binaryReader = new BinaryReader(memStream))
                {
                    while (memStream.Position < memStream.Length)
                    {
                        Guid channelId = Guid.Empty;
                        byte[] message = null;
                        try
                        {
                            channelId = new Guid(binaryReader.ReadBytes(16));
                            var messageLength = binaryReader.ReadInt32();
                            message = binaryReader.ReadBytes(messageLength);
                        }
                        catch (Exception ex)
                        {
                            throw new UnityAgentsException(
                                "There was a problem reading a message in a SideChannel. Please make sure the " +
                                "version of MLAgents in Unity is compatible with the Python version. Original error : "
                                + ex.Message);
                        }
                        if (sideChannels.ContainsKey(channelId))
                        {
                            using (var incomingMsg = new IncomingMessage(message))
                            {
                                sideChannels[channelId].OnMessageReceived(incomingMsg);
                            }
                        }
                        else
                        {
                            // Don't recognize this ID, but cache it in case the SideChannel that can handle
                            // it is registered before the next call to ProcessSideChannelData.
                            m_CachedMessages.Enqueue(new CachedSideChannelMessage
                            {
                                ChannelId = channelId,
                                Message = message
                            });
                        }
                    }
                }
            }
        }

    }
}
