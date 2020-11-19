using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.Analytics;
using UnityEngine.Analytics;

namespace Unity.MLAgents.Analytics
{
    internal class InferenceAnalytics
    {
        static bool s_EventRegistered = false;
        const int k_MaxEventsPerHour = 1000;
        const int k_MaxNumberOfElements = 1000;
        const string k_VendorKey = "unity.ml-agents";
        const string k_EventName = "InferenceModelSet";

        static bool EnableAnalytics()
        {
            if (s_EventRegistered)
            {
                return true;
            }

            AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(k_EventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey);
            if (result == AnalyticsResult.Ok)
            {
                s_EventRegistered = true;
            }

            return s_EventRegistered;
        }

        public static void InferenceModelSet(
            NNModel nnModel,
            string behaviorName,
            InferenceDevice inferenceDevice,
            IList<ISensor> sensors,
            ActionSpec actionSpec
        )
        {
            // The event shouldn't be able to report if this is disabled but if we know we're not going to report
            // Lets early out and not waste time gathering all the data
            if (!EditorAnalytics.enabled)
                return;

            if (!EnableAnalytics())
                return;

            var data = GetEventForModel(nnModel, behaviorName, inferenceDevice, sensors, actionSpec);
            //EditorAnalytics.SendEventWithLimit(k_EventName, data);
        }

        static InferenceEvent GetEventForModel(
            NNModel nnModel,
            string behaviorName,
            InferenceDevice inferenceDevice,
            IList<ISensor> sensors,
            ActionSpec actionSpec
        )
        {
            var barracudaModel = ModelLoader.Load(nnModel);
            var inferenceEvent = new InferenceEvent();
            inferenceEvent.BehaviorName = behaviorName;
            inferenceEvent.BarracudaModelSource = barracudaModel.IrSource;
            inferenceEvent.BarracudaModelVersion = barracudaModel.IrVersion;
            inferenceEvent.BarracudaModelProducer = barracudaModel.ProducerName;
            inferenceEvent.InferenceDevice = (int)inferenceDevice;

            if (barracudaModel.ProducerName == "Script")
            {
                // .nn files don't have these fields set correctly. Assign some placeholder values.
                inferenceEvent.BarracudaModelSource = "NN";
                inferenceEvent.BarracudaModelProducer = "tf2bc.py";
            }

#if UNITY_2019_3_OR_NEWER
            var barracudaPackageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Tensor).Assembly);
            inferenceEvent.BarracudaPackageVersion = barracudaPackageInfo.version;
#else
            inferenceEvent.BarracudaPackageVersion = "unknown";
#endif

            inferenceEvent.ActionSpec = EventActionSpec.FromActionSpec(actionSpec);
            inferenceEvent.ObservationSpecs = new List<EventObservationSpec>(sensors.Count);
            foreach (var sensor in sensors)
            {
                inferenceEvent.ObservationSpecs.Add(EventObservationSpec.FromSensor(sensor));
            }

            return inferenceEvent;
        }
    }
}
