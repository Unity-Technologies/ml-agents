using System;
using System.Collections.Generic;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Analytics
{
    internal struct InferenceEvent
    {
        /// <summary>
        /// Hash of the BehaviorName.
        /// </summary>
        public string BehaviorName;
        public string BarracudaModelSource;
        public string BarracudaModelVersion;
        public string BarracudaModelProducer;
        public string BarracudaPackageVersion;
        /// <summary>
        /// Whether inference is performed on CPU (0) or GPU (1).
        /// </summary>
        public int InferenceDevice;
        public List<EventObservationSpec> ObservationSpecs;
        public EventActionSpec ActionSpec;
        public int MemorySize;
        public long TotalWeightSizeBytes;
        public string ModelHash;
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

        public static EventActionSpec FromBrainParameters(BrainParameters brainParameters)
        {
            if (brainParameters.VectorActionSpaceType == SpaceType.Continuous)
            {
                return new EventActionSpec
                {
                    NumContinuousActions = brainParameters.NumActions,
                    NumDiscreteActions = 0,
                    BranchSizes = Array.Empty<int>(),
                };
            }
            else
            {
                return new EventActionSpec
                {
                    NumContinuousActions = 0,
                    NumDiscreteActions = brainParameters.NumActions,
                    BranchSizes = brainParameters.VectorActionSize,
                };
            }
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
        public EventObservationDimensionInfo[] DimensionInfos;

        public static EventObservationSpec FromSensor(ISensor sensor)
        {
            var shape = sensor.GetObservationShape();
            var dimInfos = new EventObservationDimensionInfo[shape.Length];
            for (var i = 0; i < shape.Length; i++)
            {
                dimInfos[i].Size = shape[i];
                // TODO copy flags when we have them
            }

            var builtInSensorType = sensor.GetBuiltInSensorType();

            return new EventObservationSpec
            {
                SensorName = sensor.GetName(),
                CompressionType = sensor.GetCompressionType().ToString(),
                BuiltInSensorType = (int)builtInSensorType,
                DimensionInfos = dimInfos,
            };
        }
    }

    internal struct RemotePolicyInitializedEvent
    {
        public string TrainingSessionGuid;
        /// <summary>
        /// Hash of the BehaviorName.
        /// </summary>
        public string BehaviorName;
        public List<EventObservationSpec> ObservationSpecs;
        public EventActionSpec ActionSpec;

        /// <summary>
        /// This will be the same as TrainingEnvironmentInitializedEvent if available, but
        /// TrainingEnvironmentInitializedEvent maybe not always be available with older trainers.
        /// </summary>
        public string MLAgentsEnvsVersion;
        public string TrainerCommunicationVersion;
    }

    // These were added as part of a new interface in https://github.com/Unity-Technologies/ml-agents/pull/4871/
    // Since we can't add a new interface in a patch release, we'll detect the type of the sensor and return
    // the enum accordingly
    internal enum BuiltInSensorType
    {
        Unknown = 0,
        VectorSensor = 1,
        // Note that StackingSensor actually returns the wrapped sensor's type
        StackingSensor = 2,
        RayPerceptionSensor = 3,
        // ReflectionSensor = 4, // Added after 1.0.x
        CameraSensor = 5,
        RenderTextureSensor = 6,
        // BufferSensor = 7, // Added after 1.0.x
        // PhysicsBodySensor = 8, // In extensions package
        // Match3Sensor = 9,  // In extensions package
        // GridSensor = 10  // In extensions package
    }

    /// <summary>
    /// Helper methods to be shared by all classes that implement <see cref="ISensor"/>.
    /// </summary>
    internal static class BuiltInSensorExtensions
    {
        /// <summary>
        /// Get the total number of elements in the ISensor's observation (i.e. the product of the
        /// shape elements).
        /// </summary>
        /// <param name="sensor"></param>
        /// <returns></returns>
        public static BuiltInSensorType GetBuiltInSensorType(this ISensor sensor)
        {
            if (sensor as VectorSensor != null)
            {
                return BuiltInSensorType.VectorSensor;
            }
            if (sensor as RayPerceptionSensor != null)
            {
                return BuiltInSensorType.RayPerceptionSensor;
            }
            if (sensor as CameraSensor != null)
            {
                return BuiltInSensorType.CameraSensor;
            }
            if (sensor as RenderTextureSensor != null)
            {
                return BuiltInSensorType.RenderTextureSensor;
            }
            var stackingSensor = sensor as StackingSensor;
            if (stackingSensor != null)
            {
                // Recurse on the wrapped sensor
                return stackingSensor.GetWrappedSensor().GetBuiltInSensorType() ;
            }
            return BuiltInSensorType.Unknown;
        }
    }
}
