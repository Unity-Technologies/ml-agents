using System;
using UnityEngine;
using Unity.MLAgents.Analytics;
using Unity.MLAgents.CommunicatorObjects;

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

            if (anyMessage.Is(TrainingEnvironmentInitialized.Descriptor))
            {
                var envInitProto = anyMessage.Unpack<TrainingEnvironmentInitialized>();
                var envInitEvent = envInitProto.ToTrainingEnvironmentInitializedEvent();
                TrainingAnalytics.TrainingEnvironmentInitialized(envInitEvent);
            }
            else if (anyMessage.Is(TrainingBehaviorInitialized.Descriptor))
            {
                var behaviorInitProto = anyMessage.Unpack<TrainingBehaviorInitialized>();
                var behaviorTrainingEvent = behaviorInitProto.ToTrainingBehaviorInitializedEvent();
                TrainingAnalytics.TrainingBehaviorInitialized(behaviorTrainingEvent);
            }
            else
            {
                // TODO WARN?
            }
        }
    }
}
