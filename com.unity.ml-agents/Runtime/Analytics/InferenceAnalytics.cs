using System;
using System.Collections.Generic;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.Analytics;
using UnityEngine;
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

        private static HashSet<NNModel> s_SentModels;

        static bool EnableAnalytics()
        {
            if (s_EventRegistered)
            {
                return true;
            }

            if (s_SentModels == null)
            {
                s_SentModels = new HashSet<NNModel>();
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

            var added = s_SentModels.Add(nnModel);

            if (!added)
            {
                // We previously added this model. Exit so we don't resend.
                return;
            }

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
            inferenceEvent.MemorySize = (int)barracudaModel.GetTensorByName(TensorNames.MemorySize)[0];
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

            inferenceEvent.ModelHash = GetModelHash(barracudaModel);
            return inferenceEvent;
        }

        internal class FNVHash
        {
            const ulong kFNV_prime = 1099511628211;
            const ulong kFNV_offset_basis = 14695981039346656037;
            private const int kMaxBytes = 1024;

            public ulong hash;

            public FNVHash()
            {
                hash = kFNV_offset_basis;
            }

            public void Append(float[] values)
            {
                // Limit the max number of float bytes that we hash for performance.
                // This increases the chance of a collision, but this should still be extremely rare.
                var bytesToHash = Mathf.Min(kMaxBytes, Buffer.ByteLength(values));
                for (var i = 0; i < bytesToHash; i++)
                {
                    var b = Buffer.GetByte(values, i);
                    Update(b);
                }
            }

            public void Append(string value)
            {
                foreach (var c in value)
                {
                    Update((byte)c);
                }
            }

            private void Update(byte b)
            {
                hash *= kFNV_prime;
                hash ^= b;
            }

            public override string ToString()
            {
                return hash.ToString();
            }
        }

        static string GetModelHash(Model barracudaModel)
        {
            // Pre-2020 versions of Unity don't have Hash128.Append() (can only hash strings)
            // For these versions, we'll use a simple FNV-1 hash.
            // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
#if UNITY_2020_1_OR_NEWER
            var hash = new Hash128();
#else
            var hash = new FNVHash();
#endif
            foreach (var layer in barracudaModel.layers)
            {
                hash.Append(layer.name);
                hash.Append(layer.weights);
            }

            return hash.ToString();
        }
    }
}
