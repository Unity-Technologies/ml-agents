using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Analytics;

namespace Unity.MLAgents.Analytics
{
    internal static class AnalyticsConstants
    {
        public const string k_VendorKey = "unity.ml-agents";

        /// <summary>
        /// Maximum number of events sent per hour.
        /// </summary>
        public const int k_MaxEventsPerHour = 1000;

        /// <summary>
        /// Maximum number of items in an event.
        /// </summary>
        public const int k_MaxNumberOfElements = 1000;
    }

    [Serializable]
    [AnalyticInfo(eventName: "ml_agents_inferencemodelset", version: 1, vendorKey: AnalyticsConstants.k_VendorKey, maxEventsPerHour: AnalyticsConstants.k_MaxEventsPerHour, maxNumberOfElements: AnalyticsConstants.k_MaxNumberOfElements)]
    internal class InferenceEvent : IAnalytic.IData, IAnalytic
    {
        /// <summary>
        /// Hash of the BehaviorName.
        /// </summary>
        public string BehaviorName;
        public string SentisModelSource;
        public long SentisModelVersion;
        public string SentisModelProducer;
        public string SentisPackageVersion;
        /// <summary>
        /// Whether inference is performed on CPU (0) or GPU (1).
        /// </summary>
        public int InferenceDevice;
        public List<EventObservationSpec> ObservationSpecs;
        public EventActionSpec ActionSpec;
        public List<EventActuatorInfo> ActuatorInfos;
        public int MemorySize;
        public long TotalWeightSizeBytes;
        public string ModelHash;
        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            data = this;
            error = null;
            return true;
        }
    }

    /// <summary>
    /// Simplified version of ActionSpec struct for use in analytics
    /// </summary>
    [Serializable]
    internal struct EventActionSpec
    {
        public int NumContinuousActions;
        public int NumDiscreteActions;
        public int[] BranchSizes;

        public static EventActionSpec FromActionSpec(ActionSpec actionSpec)
        {
            var branchSizes = actionSpec.BranchSizes ?? Array.Empty<int>();
            return new EventActionSpec
            {
                NumContinuousActions = actionSpec.NumContinuousActions,
                NumDiscreteActions = actionSpec.NumDiscreteActions,
                BranchSizes = branchSizes,
            };
        }
    }

    /// <summary>
    /// Information about an actuator.
    /// </summary>
    [Serializable]
    internal struct EventActuatorInfo
    {
        public int BuiltInActuatorType;
        public int NumContinuousActions;
        public int NumDiscreteActions;

        public static EventActuatorInfo FromActuator(IActuator actuator)
        {
            BuiltInActuatorType builtInActuatorType = Actuators.BuiltInActuatorType.Unknown;
            if (actuator is IBuiltInActuator builtInActuator)
            {
                builtInActuatorType = builtInActuator.GetBuiltInActuatorType();
            }

            var actionSpec = actuator.ActionSpec;

            return new EventActuatorInfo
            {
                BuiltInActuatorType = (int)builtInActuatorType,
                NumContinuousActions = actionSpec.NumContinuousActions,
                NumDiscreteActions = actionSpec.NumDiscreteActions
            };
        }
    }

    /// <summary>
    /// Information about one dimension of an observation.
    /// </summary>
    [Serializable]
    internal struct EventObservationDimensionInfo
    {
        public int Size;
        public int Flags;
    }

    /// <summary>
    /// Simplified summary of Agent observations for use in analytics
    /// </summary>
    [Serializable]
    internal struct EventObservationSpec
    {
        public string SensorName;
        public string CompressionType;
        public int BuiltInSensorType;
        public int ObservationType;
        public EventObservationDimensionInfo[] DimensionInfos;

        public static EventObservationSpec FromSensor(ISensor sensor)
        {
            var obsSpec = sensor.GetObservationSpec();
            var shape = obsSpec.Shape;
            var dimProps = obsSpec.DimensionProperties;
            var dimInfos = new EventObservationDimensionInfo[shape.Length];
            for (var i = 0; i < shape.Length; i++)
            {
                dimInfos[i].Size = shape[i];
                dimInfos[i].Flags = (int)dimProps[i];
            }

            var builtInSensorType =
                (sensor as IBuiltInSensor)?.GetBuiltInSensorType() ?? Sensors.BuiltInSensorType.Unknown;

            return new EventObservationSpec
            {
                SensorName = sensor.GetName(),
                CompressionType = sensor.GetCompressionSpec().SensorCompressionType.ToString(),
                BuiltInSensorType = (int)builtInSensorType,
                ObservationType = (int)obsSpec.ObservationType,
                DimensionInfos = dimInfos,
            };
        }
    }

    [Serializable]
    [AnalyticInfo(eventName: "ml_agents_remote_policy_initialized", vendorKey: AnalyticsConstants.k_VendorKey, maxEventsPerHour: AnalyticsConstants.k_MaxEventsPerHour, maxNumberOfElements: AnalyticsConstants.k_MaxNumberOfElements)]
    internal class RemotePolicyInitializedEvent : IAnalytic.IData, IAnalytic
    {
        public string TrainingSessionGuid;
        /// <summary>
        /// Hash of the BehaviorName.
        /// </summary>
        public string BehaviorName;
        public List<EventObservationSpec> ObservationSpecs;
        public EventActionSpec ActionSpec;
        public List<EventActuatorInfo> ActuatorInfos;

        /// <summary>
        /// This will be the same as TrainingEnvironmentInitializedEvent if available, but
        /// TrainingEnvironmentInitializedEvent maybe not always be available with older trainers.
        /// </summary>
        public string MLAgentsEnvsVersion;
        public string TrainerCommunicationVersion;
        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            data = this;
            error = null;
            return true;
        }
    }


    [Serializable]
    [AnalyticInfo(eventName: "ml_agents_training_environment_initialized", vendorKey: AnalyticsConstants.k_VendorKey, maxEventsPerHour: AnalyticsConstants.k_MaxEventsPerHour, maxNumberOfElements: AnalyticsConstants.k_MaxNumberOfElements)]
    internal class TrainingEnvironmentInitializedEvent : IAnalytic.IData, IAnalytic
    {
        public string TrainingSessionGuid;

        public string TrainerPythonVersion;
        public string MLAgentsVersion;
        public string MLAgentsEnvsVersion;
        public string TorchVersion;
        public string TorchDeviceType;
        public int NumEnvironments;
        public int NumEnvironmentParameters;
        public string RunOptions;
        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            data = this;
            error = null;
            return true;
        }
    }

    [Flags]
    internal enum RewardSignals
    {
        Extrinsic = 1 << 0,
        Gail = 1 << 1,
        Curiosity = 1 << 2,
        Rnd = 1 << 3,
    }

    [Flags]
    internal enum TrainingFeatures
    {
        BehavioralCloning = 1 << 0,
        Recurrent = 1 << 1,
        Threaded = 1 << 2,
        SelfPlay = 1 << 3,
        Curriculum = 1 << 4,
    }

    [Serializable]
    [AnalyticInfo(eventName: "ml_agents_training_behavior_initialized", vendorKey: AnalyticsConstants.k_VendorKey, maxEventsPerHour: AnalyticsConstants.k_MaxEventsPerHour, maxNumberOfElements: AnalyticsConstants.k_MaxNumberOfElements)]
    internal class TrainingBehaviorInitializedEvent : IAnalytic.IData, IAnalytic
    {
        public string TrainingSessionGuid;

        public string BehaviorName;
        public string TrainerType;
        public RewardSignals RewardSignalFlags;
        public TrainingFeatures TrainingFeatureFlags;
        public string VisualEncoder;
        public int NumNetworkLayers;
        public int NumNetworkHiddenUnits;
        public string Config;

        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            data = this;
            error = null;
            return true;
        }
    }
}
