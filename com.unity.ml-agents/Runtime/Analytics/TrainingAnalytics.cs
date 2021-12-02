using System;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if MLA_UNITY_ANALYTICS_MODULE

#if ENABLE_CLOUD_SERVICES_ANALYTICS
using UnityEngine.Analytics;
#endif

#if UNITY_EDITOR
using UnityEditor.Analytics;
#endif
#endif

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Analytics
{
    internal static class TrainingAnalytics
    {
        const string k_VendorKey = "unity.ml-agents";
        const string k_TrainingEnvironmentInitializedEventName = "ml_agents_training_environment_initialized";
        const string k_TrainingBehaviorInitializedEventName = "ml_agents_training_behavior_initialized";
        const string k_RemotePolicyInitializedEventName = "ml_agents_remote_policy_initialized";

        private static readonly string[] s_EventNames =
        {
            k_TrainingEnvironmentInitializedEventName,
            k_TrainingBehaviorInitializedEventName,
            k_RemotePolicyInitializedEventName
        };

        /// <summary>
        /// Hourly limit for this event name
        /// </summary>
        const int k_MaxEventsPerHour = 1000;

        /// <summary>
        /// Maximum number of items in this event.
        /// </summary>
        const int k_MaxNumberOfElements = 1000;

        private static bool s_SentEnvironmentInitialized;

#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
        /// <summary>
        /// Whether or not we've registered this particular event yet
        /// </summary>
        static bool s_EventsRegistered;

        /// <summary>
        /// Behaviors that we've already sent events for.
        /// </summary>
        private static HashSet<string> s_SentRemotePolicyInitialized;
        private static HashSet<string> s_SentTrainingBehaviorInitialized;
#endif

        private static Guid s_TrainingSessionGuid;

        // These are set when the RpcCommunicator connects
        private static string s_TrainerPackageVersion = "";
        private static string s_TrainerCommunicationVersion = "";

        internal static bool EnableAnalytics()
        {
#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
            if (s_EventsRegistered)
            {
                return true;
            }
            foreach (var eventName in s_EventNames)
            {
                AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(eventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey);
                if (result != AnalyticsResult.Ok)
                {
                    return false;
                }
            }
            s_EventsRegistered = true;

            if (s_SentRemotePolicyInitialized == null)
            {
                s_SentRemotePolicyInitialized = new HashSet<string>();
                s_SentTrainingBehaviorInitialized = new HashSet<string>();
                s_TrainingSessionGuid = Guid.NewGuid();
            }

            return s_EventsRegistered;
#else
            return false;
#endif // MLA_UNITY_ANALYTICS_MODULE
        }

        /// <summary>
        /// Cache information about the trainer when it becomes available in the RpcCommunicator.
        /// </summary>
        /// <param name="communicationVersion"></param>
        /// <param name="packageVersion"></param>
        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        public static void SetTrainerInformation(string packageVersion, string communicationVersion)
        {
            s_TrainerPackageVersion = packageVersion;
            s_TrainerCommunicationVersion = communicationVersion;
        }

        public static bool IsAnalyticsEnabled()
        {
#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
            return EditorAnalytics.enabled;
#else
            return false;
#endif
        }

        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        public static void TrainingEnvironmentInitialized(TrainingEnvironmentInitializedEvent tbiEvent)
        {
            if (!IsAnalyticsEnabled())
                return;

            if (!EnableAnalytics())
                return;

            if (s_SentEnvironmentInitialized)
            {
                // We already sent an TrainingEnvironmentInitializedEvent. Exit so we don't resend.
                return;
            }

            s_SentEnvironmentInitialized = true;
            tbiEvent.TrainingSessionGuid = s_TrainingSessionGuid.ToString();

            // Note - to debug, use JsonUtility.ToJson on the event.
            // Debug.Log(
            //     $"Would send event {k_TrainingEnvironmentInitializedEventName} with body {JsonUtility.ToJson(tbiEvent, true)}"
            // );
#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
            if (AnalyticsUtils.s_SendEditorAnalytics)
            {
                EditorAnalytics.SendEventWithLimit(k_TrainingEnvironmentInitializedEventName, tbiEvent);
            }
#endif
        }

        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        public static void RemotePolicyInitialized(
            string fullyQualifiedBehaviorName,
            IList<ISensor> sensors,
            ActionSpec actionSpec,
            IList<IActuator> actuators
        )
        {
#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
            if (!IsAnalyticsEnabled())
                return;

            if (!EnableAnalytics())
                return;

            // Extract base behavior name (no team ID)
            var behaviorName = ParseBehaviorName(fullyQualifiedBehaviorName);
            var added = s_SentRemotePolicyInitialized.Add(behaviorName);

            if (!added)
            {
                // We previously added this model. Exit so we don't resend.
                return;
            }

            var data = GetEventForRemotePolicy(behaviorName, sensors, actionSpec, actuators);
            // Note - to debug, use JsonUtility.ToJson on the event.
            // Debug.Log(
            //     $"Would send event {k_RemotePolicyInitializedEventName} with body {JsonUtility.ToJson(data, true)}"
            // );
            if (AnalyticsUtils.s_SendEditorAnalytics)
            {
                EditorAnalytics.SendEventWithLimit(k_RemotePolicyInitializedEventName, data);
            }
#endif
        }

        internal static string ParseBehaviorName(string fullyQualifiedBehaviorName)
        {
            var lastQuestionIndex = fullyQualifiedBehaviorName.LastIndexOf("?");
            if (lastQuestionIndex < 0)
            {
                // Nothing to remove
                return fullyQualifiedBehaviorName;
            }

            return fullyQualifiedBehaviorName.Substring(0, lastQuestionIndex);
        }

        internal static TrainingBehaviorInitializedEvent SanitizeTrainingBehaviorInitializedEvent(TrainingBehaviorInitializedEvent tbiEvent)
        {
            // Hash the behavior name if the message version is from an older version of ml-agents that doesn't do trainer-side hashing.
            // We'll also, for extra safety, verify that the BehaviorName is the size of the expected SHA256 hash.
            // Context: The config field was added at the same time as trainer side hashing, so messages including it should already be hashed.
            if (tbiEvent.Config.Length == 0 || tbiEvent.BehaviorName.Length != 64)
            {
                tbiEvent.BehaviorName = AnalyticsUtils.Hash(k_VendorKey, tbiEvent.BehaviorName);
            }

            return tbiEvent;
        }

        [Conditional("MLA_UNITY_ANALYTICS_MODULE")]
        public static void TrainingBehaviorInitialized(TrainingBehaviorInitializedEvent rawTbiEvent)
        {
#if UNITY_EDITOR && MLA_UNITY_ANALYTICS_MODULE && ENABLE_CLOUD_SERVICES_ANALYTICS
            if (!IsAnalyticsEnabled())
                return;

            if (!EnableAnalytics())
                return;

            var tbiEvent = SanitizeTrainingBehaviorInitializedEvent(rawTbiEvent);
            var behaviorName = tbiEvent.BehaviorName;
            var added = s_SentTrainingBehaviorInitialized.Add(behaviorName);

            if (!added)
            {
                // We previously added this model. Exit so we don't resend.
                return;
            }

            tbiEvent.TrainingSessionGuid = s_TrainingSessionGuid.ToString();

            // Note - to debug, use JsonUtility.ToJson on the event.
            // Debug.Log(
            //     $"Would send event {k_TrainingBehaviorInitializedEventName} with body {JsonUtility.ToJson(tbiEvent, true)}"
            // );
            if (AnalyticsUtils.s_SendEditorAnalytics)
            {
                EditorAnalytics.SendEventWithLimit(k_TrainingBehaviorInitializedEventName, tbiEvent);
            }
#endif
        }

        internal static RemotePolicyInitializedEvent GetEventForRemotePolicy(
            string behaviorName,
            IList<ISensor> sensors,
            ActionSpec actionSpec,
            IList<IActuator> actuators
        )
        {
            var remotePolicyEvent = new RemotePolicyInitializedEvent();

            // Hash the behavior name so that there's no concern about PII or "secret" data being leaked.
            remotePolicyEvent.BehaviorName = AnalyticsUtils.Hash(k_VendorKey, behaviorName);

            remotePolicyEvent.TrainingSessionGuid = s_TrainingSessionGuid.ToString();
            remotePolicyEvent.ActionSpec = EventActionSpec.FromActionSpec(actionSpec);
            remotePolicyEvent.ObservationSpecs = new List<EventObservationSpec>(sensors.Count);
            foreach (var sensor in sensors)
            {
                remotePolicyEvent.ObservationSpecs.Add(EventObservationSpec.FromSensor(sensor));
            }

            remotePolicyEvent.ActuatorInfos = new List<EventActuatorInfo>(actuators.Count);
            foreach (var actuator in actuators)
            {
                remotePolicyEvent.ActuatorInfos.Add(EventActuatorInfo.FromActuator(actuator));
            }

            remotePolicyEvent.MLAgentsEnvsVersion = s_TrainerPackageVersion;
            remotePolicyEvent.TrainerCommunicationVersion = s_TrainerCommunicationVersion;
            return remotePolicyEvent;
        }
    }
}
