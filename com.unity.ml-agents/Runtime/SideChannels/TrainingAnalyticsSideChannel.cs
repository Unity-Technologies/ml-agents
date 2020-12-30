using System;
using UnityEngine;
using Unity.MLAgents.Analytics;

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
            // TODO logic here is placeholder until we define protos for sending information.
            var eventName = msg.ReadString();
            if (eventName == "environment_initialized")
            {
                var eventBody = msg.ReadString();
                Debug.Log($"{eventName}: {eventBody}");

                var envInitEvent = new TrainingEnvironmentInitializedEvent
                {

                };
                TrainingAnalytics.TrainingEnvironmentInitialized(envInitEvent);
            }
            else // TrainingBehaviorInitialized
            {
                var behaviorName = msg.ReadString();
                var eventBody = msg.ReadString();
                Debug.Log($"{eventName} ({behaviorName}): {eventBody}");

                var behaviorTrainingEvent = new TrainingBehaviorInitializedEvent
                {
                    BehaviorName = behaviorName,
                };
                TrainingAnalytics.TrainingBehaviorInitialized(behaviorTrainingEvent);
            }
        }
    }
}
