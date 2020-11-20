using System;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Analytics
{
    internal struct InferenceEvent
    {
        public string BehaviorName;
        public string BarracudaModelSource;
        public string BarracudaModelVersion;
        public string BarracudaModelProducer;
        public string BarracudaPackageVersion;
        public int InferenceDevice;
        public List<EventObservationSpec> ObservationSpecs;
        public EventActionSpec ActionSpec;
        public int MemorySize;
        public string ModelHash;
    }

    /// <summary>
    /// Simplified version of ActionSpec struct for use in analytics
    /// </summary>
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
    /// Simplified summary of Agent observations for use in analytics
    /// </summary>
    internal struct EventObservationSpec
    {
        public string SensorName;
        public int[] ObservationShape;

        public static EventObservationSpec FromSensor(ISensor sensor)
        {
            return new EventObservationSpec
            {
                SensorName = sensor.GetName(),
                ObservationShape = sensor.GetObservationShape(),
            };
        }
    }
}
