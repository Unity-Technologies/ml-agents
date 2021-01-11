using System;
using UnityEngine;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.CommunicatorObjects;
using static Google.Protobuf.WellKnownTypes;

namespace Unity.MLAgents.SideChannels
{
    public class TrainingAnalyticsSideChannel : SideChannel
    {
        const string k_TrainingAnalyticsConfigId = "b664a4a9-d86f-5a5f-95cb-e8353a7e8356";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal TrainingAnalyticsSideChannel()
        {
            ChannelId = new Guid(k_TrainingAnalyticsConfigId);
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            var anyMessage = Google.Protobuf.WellKnownTypes.Any.Parser.ParseFrom(msg.GetRawBytes());
            var envInitProto = anyMessage.Unpack<TrainingEnvironmentInitialized>();
            var behaviorInitProto = anyMessage.Unpack<TrainingBehaviorInitialized>();

            if (envInitProto != null)
            {

                Debug.Log($"envInitProto init: {envInitProto}");

                var envInitEvent = new TrainingEnvironmentInitializedEvent
                {

                };
                TrainingAnalytics.TrainingEnvironmentInitialized(envInitEvent);
            }
            else // TrainingBehaviorInitialized
            {
                Debug.Log($"behaviorInitProto ({behaviorInitProto.BehaviorName}): {behaviorInitProto}");

                var behaviorTrainingEvent = new TrainingBehaviorInitializedEvent
                {
                    BehaviorName = behaviorInitProto.BehaviorName,
                };
                TrainingAnalytics.TrainingBehaviorInitialized(behaviorTrainingEvent);
            }
        }
    }
}
