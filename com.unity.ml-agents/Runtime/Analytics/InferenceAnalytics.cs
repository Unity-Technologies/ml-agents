using System;
using System.Collections.Generic;
using Unity.Barracuda;
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
        const string k_VendorKey = "unity.ml-agents";
        const string k_EventName = "ml_agents_inferencemodelset";
        const int k_EventVersion = 1;

        /// <summary>
        /// Whether or not we've registered this particular event yet
        /// </summary>
        static bool s_EventRegistered = false;

        /// <summary>
        /// Hourly limit for this event name
        /// </summary>
        const int k_MaxEventsPerHour = 1000;

        /// <summary>
        /// Maximum number of items in this event.
        /// </summary>
        const int k_MaxNumberOfElements = 1000;


        /// <summary>
        /// Models that we've already sent events for.
        /// </summary>
        private static HashSet<NNModel> s_SentModels;

        static bool EnableAnalytics()
        {
            if (s_EventRegistered)
            {
                return true;
            }

#if UNITY_EDITOR
            AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(k_EventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey, k_EventVersion);
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

        /// <summary>
        /// Send an analytics event for the NNModel when it is set up for inference.
        /// No events will be sent if analytics are disabled, and at most one event
        /// will be sent per model instance.
        /// </summary>
        /// <param name="nnModel">The NNModel being used for inference.</param>
        /// <param name="behaviorName">The BehaviorName of the Agent using the model</param>
        /// <param name="inferenceDevice">Whether inference is being performed on the CPU or GPU</param>
        /// <param name="sensors">List of ISensors for the Agent. Used to generate information about the observation space.</param>
        /// <param name="brainParameters">BrainParameters for the Agent. Used to generate information about the action space.</param>
        /// <returns></returns>
        public static void InferenceModelSet(
            NNModel nnModel,
            string behaviorName,
            InferenceDevice inferenceDevice,
            IList<ISensor> sensors,
            BrainParameters brainParameters
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

            var data = GetEventForModel(nnModel, behaviorName, inferenceDevice, sensors, brainParameters);
            // Note - to debug, use JsonUtility.ToJson on the event.
            //Debug.Log(JsonUtility.ToJson(data, true));
#if UNITY_EDITOR
            if (AnalyticsUtils.s_SendEditorAnalytics)
            {
                EditorAnalytics.SendEventWithLimit(k_EventName, data, k_EventVersion);
            }
#else
            return;
#endif
        }

        /// <summary>
        /// Generate an InferenceEvent for the model.
        /// </summary>
        /// <param name="nnModel"></param>
        /// <param name="behaviorName"></param>
        /// <param name="inferenceDevice"></param>
        /// <param name="sensors"></param>
        /// <param name="actionSpec"></param>
        /// <returns></returns>
        internal static InferenceEvent GetEventForModel(
            NNModel nnModel,
            string behaviorName,
            InferenceDevice inferenceDevice,
            IList<ISensor> sensors,
            BrainParameters brainParameters
        )
        {
            var barracudaModel = ModelLoader.Load(nnModel);
            var inferenceEvent = new InferenceEvent();

            // Hash the behavior name so that there's no concern about PII or "secret" data being leaked.
            inferenceEvent.BehaviorName = AnalyticsUtils.Hash(behaviorName);

            inferenceEvent.BarracudaModelSource = barracudaModel.IrSource;
            inferenceEvent.BarracudaModelVersion = barracudaModel.IrVersion;
            inferenceEvent.BarracudaModelProducer = barracudaModel.ProducerName;
            inferenceEvent.MemorySize = (int)barracudaModel.GetTensorByName(TensorNames.MemorySize)[0];
            inferenceEvent.InferenceDevice = (int)inferenceDevice;

            if (barracudaModel.ProducerName == "Script")
            {
                // .nn files don't have these fields set correctly. Assign some placeholder values.
                inferenceEvent.BarracudaModelSource = "NN";
                inferenceEvent.BarracudaModelProducer = "tensorflow_to_barracuda.py";
            }

#if UNITY_2019_3_OR_NEWER && UNITY_EDITOR
            var barracudaPackageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(Tensor).Assembly);
            inferenceEvent.BarracudaPackageVersion = barracudaPackageInfo.version;
#else
            inferenceEvent.BarracudaPackageVersion = null;
#endif

            inferenceEvent.ActionSpec = EventActionSpec.FromBrainParameters(brainParameters);
            inferenceEvent.ObservationSpecs = new List<EventObservationSpec>(sensors.Count);
            foreach (var sensor in sensors)
            {
                inferenceEvent.ObservationSpecs.Add(EventObservationSpec.FromSensor(sensor));
            }

            inferenceEvent.TotalWeightSizeBytes = GetModelWeightSize(barracudaModel);
            inferenceEvent.ModelHash = GetModelHash(barracudaModel);
            return inferenceEvent;
        }

        /// <summary>
        /// Compute the total model weight size in bytes.
        /// This corresponds to the "Total weight size" display in the Barracuda inspector,
        /// and the calculations are the same.
        /// </summary>
        /// <param name="barracudaModel"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Wrapper around Hash128 that supports Append(float[], int, int)
        /// </summary>
        struct MLAgentsHash128
        {
            private Hash128 m_Hash;

            public void Append(float[] values, int count)
            {
                if (values == null)
                {
                    return;
                }

                // Pre-2020 versions of Unity don't have Hash128.Append() (can only hash strings and scalars)
                // For these versions, we'll hash element by element.
#if UNITY_2020_1_OR_NEWER
                m_Hash.Append(values, 0, count);
#else
                for (var i = 0; i < count; i++)
                {
                    var tempHash = new Hash128();
                    HashUtilities.ComputeHash128(ref values[i], ref tempHash);
                    HashUtilities.AppendHash(ref tempHash, ref m_Hash);
                }
#endif
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

        /// <summary>
        /// Compute a hash of the model's layer data and return it as a string.
        /// A subset of the layer weights are used for performance.
        /// This increases the chance of a collision, but this should still be extremely rare.
        /// </summary>
        /// <param name="barracudaModel"></param>
        /// <returns></returns>
        static string GetModelHash(Model barracudaModel)
        {
            var hash = new MLAgentsHash128();

            // Limit the max number of float bytes that we hash for performance.
            const int kMaxFloats = 256;

            foreach (var layer in barracudaModel.layers)
            {
                hash.Append(layer.name);
                var numFloatsToHash = Mathf.Min(layer.weights.Length, kMaxFloats);
                hash.Append(layer.weights, numFloatsToHash);
            }

            return hash.ToString();
        }
    }
}
