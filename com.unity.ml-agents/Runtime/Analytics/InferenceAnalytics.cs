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
    internal class InferenceAnalytics
    {
        static bool s_EventRegistered = false;
        const int k_MaxEventsPerHour = 1000;
        const int k_MaxNumberOfElements = 1000;
        const string k_VendorKey = "unity.ml-agents";
        const string k_EventName = "ml_agents_inferencemodelset";

        private static HashSet<NNModel> s_SentModels;

        static bool EnableAnalytics()
        {
            if (s_EventRegistered)
            {
                return true;
            }

#if UNITY_EDITOR
            AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(k_EventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey);
#else
            AnalyticsResult result = AnalyticsResult.UnsupportedPlatform;
#endif
            if (result == AnalyticsResult.Ok)
            {
                s_EventRegistered = true;
            }

            if (s_EventRegistered && s_SentModels == null)
            {
                s_SentModels = new HashSet<NNModel>();
            }

            return s_EventRegistered;
        }

        public static bool IsAnalyticsEnabled()
        {
#if UNITY_EDITOR
            return EditorAnalytics.enabled;
#else
            return false;
#endif
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
            if (!IsAnalyticsEnabled())
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
            // Note - to debug, use JsonUtility.ToJson on the event.
            //Debug.Log(JsonUtility.ToJson(data, true));
#if UNITY_EDITOR
            //EditorAnalytics.SendEventWithLimit(k_EventName, data);
#else
            return;
#endif
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

            inferenceEvent.TotalWeightSizeBytes = GetModelWeightSize(barracudaModel);
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

            public void Append(float[] values, int startUnused, int count)
            {
                var bytesToHash = sizeof(float) * count;
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

        struct MLAgentsHash128
        {
            private Hash128 m_Hash;

            public void Append(float[] values, int startUnused, int count)
            {
                if (values == null)
                {
                    return;
                }

                for (var i = 0; i < count; i++)
                {
                    var tempHash = new Hash128();
                    HashUtilities.ComputeHash128(ref values[i], ref tempHash);
                    HashUtilities.AppendHash(ref tempHash, ref m_Hash);
                }
            }

            public void Append(string value)
            {
                var tempHash = Hash128.Compute(value);
                HashUtilities.AppendHash(ref tempHash, ref m_Hash);
            }

            public override string ToString()
            {
                return m_Hash.ToString();
            }
        }

        static long GetModelWeightSize(Model barracudaModel)
        {
            long totalWeightsSizeInBytes = 0;
            for (var l = 0; l < barracudaModel.layers.Count; ++l)
            {
                for (var d = 0; d < barracudaModel.layers[l].datasets.Length; ++d)
                {
                    totalWeightsSizeInBytes += barracudaModel.layers[l].datasets[d].length;
                }
            }
            return totalWeightsSizeInBytes;
        }

        static string GetModelHash(Model barracudaModel)
        {
            // Pre-2020 versions of Unity don't have Hash128.Append() (can only hash strings)
            // For these versions, we'll use a simple wrapper that supports arrays of floats.
#if UNITY_2020_1_OR_NEWER
            var hash = new Hash128();
#else
            var hash = new MLAgentsHash128();
#endif
            // Limit the max number of float bytes that we hash for performance.
            // This increases the chance of a collision, but this should still be extremely rare.
            const int kMaxFloats = 256;

            foreach (var layer in barracudaModel.layers)
            {
                hash.Append(layer.name);
                var numFloatsToHash = Mathf.Min(layer.weights.Length, kMaxFloats);
                hash.Append(layer.weights, 0, numFloatsToHash);
            }

            return hash.ToString();
        }
    }
}
