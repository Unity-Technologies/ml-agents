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

        private static readonly Queue<CachedSideChannelMessage> m_CachedMessages =
            new Queue<CachedSideChannelMessage>();

        /// <summary>
        /// Register a side channel to begin sending and receiving messages. This method is
        /// available for environments that have custom side channels. All built-in side
        /// channels within the ML-Agents Toolkit are managed internally and do not need to
        /// be explicitly registered/unregistered. A side channel may only be registered once.
        /// Additionally, only one side channel of each type is allowed.
        /// </summary>
        /// <param name="sideChannel">The side channel to register.</param>
        public static void RegisterSideChannel(SideChannel sideChannel)
        {
            var channelId = sideChannel.ChannelId;
            if (RegisteredChannels.ContainsKey(channelId))
            {
                throw new UnityAgentsException(
                    $"A side channel with id {channelId} is already registered. " +
                    "You cannot register multiple side channels of the same id.");
            }
            
            // Process any messages that we've already received for this channel ID.
            var numMessages = m_CachedMessages.Count;
            for (var i = 0; i < numMessages; i++)
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
        /// Unregister a side channel to stop sending and receiving messages. This method is
        /// available for environments that have custom side channels. All built-in side
        /// channels within the ML-Agents Toolkit are managed internally and do not need to
        /// be explicitly registered/unregistered. A side channel may only be unregistered
        /// multiple times. Note that unregistering a side channel may not stop the Python side
        /// from sending them, but it does mean that sent messages with not result in a call
        /// to <see cref="SideChannel.OnMessageReceived(IncomingMessage)"/>. Furthermore,
        /// those messages will not be buffered and will, in essence, be lost.
        /// </summary>
        /// <param name="sideChannel">The side channel to unregister.</param>
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
        internal static void UnregisterAllSideChannels()
        {
            RegisteredChannels = new Dictionary<Guid, SideChannel>();
        }

        /// <summary>
        /// Returns the SideChannel of Type T if there is one registered, or null if it doesn't.
        /// If there are multiple SideChannels of the same type registered, the returned instance is arbitrary.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        internal static T GetSideChannel<T>() where T: SideChannel
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
