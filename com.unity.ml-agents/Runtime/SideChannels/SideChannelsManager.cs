using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Collection of static utilities for managing the registering/unregistering of
    /// <see cref="SideChannels"/> and the sending/receiving of messages for all the channels.
    /// </summary>
    public static class SideChannelsManager
    {
        static Dictionary<Guid, SideChannel> s_RegisteredChannels = new Dictionary<Guid, SideChannel>();

        struct CachedSideChannelMessage
        {
            public Guid ChannelId;
            public byte[] Message;
        }

        static readonly Queue<CachedSideChannelMessage> s_CachedMessages =
            new Queue<CachedSideChannelMessage>();

        /// <summary>
        /// Register a side channel to begin sending and receiving messages. This method is
        /// available for environments that have custom side channels. All built-in side
        /// channels within the ML-Agents Toolkit are managed internally and do not need to
        /// be explicitly registered/unregistered. A side channel may only be registered once.
        /// </summary>
        /// <param name="sideChannel">The side channel to register.</param>
        public static void RegisterSideChannel(SideChannel sideChannel)
        {
            var channelId = sideChannel.ChannelId;
            if (s_RegisteredChannels.ContainsKey(channelId))
            {
                throw new UnityAgentsException(
                    $"A side channel with id {channelId} is already registered. " +
                    "You cannot register multiple side channels of the same id.");
            }

            // Process any messages that we've already received for this channel ID.
            var numMessages = s_CachedMessages.Count;
            for (var i = 0; i < numMessages; i++)
            {
                var cachedMessage = s_CachedMessages.Dequeue();
                if (channelId == cachedMessage.ChannelId)
                {
                    sideChannel.ProcessMessage(cachedMessage.Message);
                }
                else
                {
                    s_CachedMessages.Enqueue(cachedMessage);
                }
            }
            s_RegisteredChannels.Add(channelId, sideChannel);
        }

        /// <summary>
        /// Unregister a side channel to stop sending and receiving messages. This method is
        /// available for environments that have custom side channels. All built-in side
        /// channels within the ML-Agents Toolkit are managed internally and do not need to
        /// be explicitly registered/unregistered. Unregistering a side channel that has already
        /// been unregistered (or never registered in the first place) has no negative side effects.
        /// Note that unregistering a side channel may not stop the Python side
        /// from sending messages, but it does mean that sent messages with not result in a call
        /// to <see cref="SideChannel.OnMessageReceived(IncomingMessage)"/>. Furthermore,
        /// those messages will not be buffered and will, in essence, be lost.
        /// </summary>
        /// <param name="sideChannel">The side channel to unregister.</param>
        public static void UnregisterSideChannel(SideChannel sideChannel)
        {
            if (s_RegisteredChannels.ContainsKey(sideChannel.ChannelId))
            {
                s_RegisteredChannels.Remove(sideChannel.ChannelId);
            }
        }

        /// <summary>
        /// Unregisters all the side channels from the communicator.
        /// </summary>
        internal static void UnregisterAllSideChannels()
        {
            s_RegisteredChannels = new Dictionary<Guid, SideChannel>();
        }

        /// <summary>
        /// Returns the SideChannel of Type T if there is one registered, or null if it doesn't.
        /// If there are multiple SideChannels of the same type registered, the returned instance is arbitrary.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        internal static T GetSideChannel<T>() where T: SideChannel
        {
            foreach (var sc in s_RegisteredChannels.Values)
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
            return GetSideChannelMessage(s_RegisteredChannels);
        }

        /// <summary>
        /// Grabs the messages that the registered side channels will send to Python at the current step
        /// into a singe byte array.
        /// </summary>
        /// <param name="sideChannels"> A dictionary of channel type to channel.</param>
        /// <returns></returns>
        internal static byte[] GetSideChannelMessage(Dictionary<Guid, SideChannel> sideChannels)
        {
            if (!HasOutgoingMessages(sideChannels))
            {
                // Early out so that we don't create the MemoryStream or BinaryWriter.
                // This is the most common case.
                return Array.Empty<byte>();
            }

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
        /// Check whether any of the sidechannels have queued messages.
        /// </summary>
        /// <param name="sideChannels"></param>
        /// <returns></returns>
        static bool HasOutgoingMessages(Dictionary<Guid, SideChannel> sideChannels)
        {
            foreach (var sideChannel in sideChannels.Values)
            {
                var messageList = sideChannel.MessageQueue;
                if (messageList.Count > 0)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Separates the data received from Python into individual messages for each registered side channel.
        /// </summary>
        /// <param name="dataReceived">The byte array of data received from Python.</param>
        internal static void ProcessSideChannelData(byte[] dataReceived)
        {
            ProcessSideChannelData(s_RegisteredChannels, dataReceived);
        }

        /// <summary>
        /// Separates the data received from Python into individual messages for each registered side channel.
        /// </summary>
        /// <param name="sideChannels">A dictionary of channel type to channel.</param>
        /// <param name="dataReceived">The byte array of data received from Python.</param>
        internal static void ProcessSideChannelData(Dictionary<Guid, SideChannel> sideChannels, byte[] dataReceived)
        {
            while (s_CachedMessages.Count != 0)
            {
                var cachedMessage = s_CachedMessages.Dequeue();
                if (sideChannels.ContainsKey(cachedMessage.ChannelId))
                {
                    sideChannels[cachedMessage.ChannelId].ProcessMessage(cachedMessage.Message);
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
                            sideChannels[channelId].ProcessMessage(message);
                        }
                        else
                        {
                            // Don't recognize this ID, but cache it in case the SideChannel that can handle
                            // it is registered before the next call to ProcessSideChannelData.
                            s_CachedMessages.Enqueue(new CachedSideChannelMessage
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
