using System;
using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Analytics;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.Analytics;
#endif

namespace Unity.MLAgents.Analytics
{
    internal class TrainingAnalytics
    {
        const string k_VendorKey = "unity.ml-agents";
        const string k_TrainingEnvironmentInitializedEventName = "ml_agents_training_environment_initialized";
        const string k_BehaviorTrainingStartedEventName = "ml_agents_training_behavior_initialized";
        const string k_RemotePolicyStartedEventName = "ml_agents_training_policy_initialized";

        private static readonly string[] s_EventNames =
        {
            k_TrainingEnvironmentInitializedEventName,
            k_BehaviorTrainingStartedEventName,
            k_RemotePolicyStartedEventName
        };

        /// <summary>
        /// Whether or not we've registered this particular event yet
        /// </summary>
        static bool s_EventsRegistered = false;

        /// <summary>
        /// Hourly limit for this event name
        /// </summary>
        const int k_MaxEventsPerHour = 1000;

        /// <summary>
        /// Maximum number of items in this event.
        /// </summary>
        const int k_MaxNumberOfElements = 1000;

        /// <summary>
        /// Behaviors that we've already sent events for.
        /// </summary>
        private static HashSet<string> s_SentRemotePolicyStartedBehaviors;

        private static Guid s_TrainingSessionId;

        static bool EnableAnalytics()
        {
            if (s_EventsRegistered)
            {
                return true;
            }

            foreach (var eventName in s_EventNames)
            {
#if UNITY_EDITOR
                AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(eventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey);
#else
                AnalyticsResult result = AnalyticsResult.UnsupportedPlatform;
#endif
                if (result != AnalyticsResult.Ok)
                {
                    return false;
                }

            }
            s_EventsRegistered = true;

            if (s_SentRemotePolicyStartedBehaviors == null)
            {
                s_SentRemotePolicyStartedBehaviors = new HashSet<string>();
                s_TrainingSessionId = Guid.NewGuid();
            }

            return s_EventsRegistered;
        }

        public static bool IsAnalyticsEnabled()
        {
#if UNITY_EDITOR
            return EditorAnalytics.enabled;
#else
            return false;
#endif
        }

        public static void RemotePolicyStarted(
            string fullyQualifiedBehaviorName,
            IList<ISensor> sensors,
            ActionSpec actionSpec
        )
        {
            // The event shouldn't be able to report if this is disabled but if we know we're not going to report
            // Lets early out and not waste time gathering all the data
            if (!IsAnalyticsEnabled())
                return;

            if (!EnableAnalytics())
                return;

            // TODO extract base behavior name (no team ID)
            var behaviorName = fullyQualifiedBehaviorName;
            var added = s_SentRemotePolicyStartedBehaviors.Add(fullyQualifiedBehaviorName);

            if (!added)
            {
                // We previously added this model. Exit so we don't resend.
                return;
            }

            var data = GetEventForRemotePolicy(behaviorName, sensors, actionSpec);
            // Note - to debug, use JsonUtility.ToJson on the event.
            Debug.Log(
                $"Would send event {k_RemotePolicyStartedEventName} with body {JsonUtility.ToJson(data, true)}"
                );
#if UNITY_EDITOR
            //EditorAnalytics.SendEventWithLimit(k_RemotePolicyStartedEventName, data);
#else
            return;
#endif
        }

        static RemotePolicyStartedEvent GetEventForRemotePolicy(
            string behaviorName,
            IList<ISensor> sensors,
            ActionSpec actionSpec)
        {
            var remotePolicyEvent = new RemotePolicyStartedEvent();

            // Hash the behavior name so that there's no concern about PII or "secret" data being leaked.
            //var behaviorNameHash = Hash128.Compute(behaviorName);
            //remotePolicyEvent.BehaviorName = behaviorNameHash.ToString();
            // TODO hash before shipping
            remotePolicyEvent.BehaviorName = behaviorName;

            remotePolicyEvent.TrainingSessionGuid = s_TrainingSessionId.ToString();
            remotePolicyEvent.ActionSpec = EventActionSpec.FromActionSpec(actionSpec);
            remotePolicyEvent.ObservationSpecs = new List<EventObservationSpec>(sensors.Count);
            foreach (var sensor in sensors)
            {
                remotePolicyEvent.ObservationSpecs.Add(EventObservationSpec.FromSensor(sensor));
            }

            return remotePolicyEvent;

        }
    }
}
