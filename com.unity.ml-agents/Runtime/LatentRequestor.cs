using System;
using Unity.MLAgents.Inference.Utils;
using Unity.MLAgents.Policies;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Unity.MLAgents
{
    [AddComponentMenu("ML Agents/LatentRequestor", (int)MenuGroup.Default)]
    [RequireComponent(typeof(Agent))]
    [DefaultExecutionOrder(-10)]
    public class LatentRequestor : MonoBehaviour
    {
        public int LatentStepsMin = 1;
        public int LatentStepsMax = 150;

        // [NonSerialized]
        Agent m_Agent;
        BehaviorParameters m_BehaviourParameters;
        private DecisionRequester m_DecisionRequestor;
        int m_LatentStepCount;
        int m_LatentSize;
        float[] m_Latents;
        RandomNormal m_RandomNormal;


        public Agent Agent => m_Agent;
        public float[] Latents => m_Latents;

        void Awake()
        {
            m_Agent = gameObject.GetComponent<Agent>();
            m_LatentStepCount = 0;
            m_BehaviourParameters = gameObject.GetComponent<BehaviorParameters>();
            m_DecisionRequestor = gameObject.GetComponent<DecisionRequester>();
            m_LatentSize = m_BehaviourParameters.BrainParameters.EmbeddingSize;
            m_RandomNormal = new RandomNormal(Academy.Instance.InferenceSeed);
            Academy.Instance.AgentPreStep += UpdateLatents;
        }

        void OnDestroy()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.AgentPreStep -= UpdateLatents;
            }
        }

        void UpdateLatents(int academyStepCount)
        {
            if (academyStepCount % m_DecisionRequestor.DecisionPeriod == 0)
            {
                if (m_LatentStepCount <= 0)
                {
                    ResetLatents();
                    ResetLatentStepCounts();
                }
                else
                {
                    m_LatentStepCount -= 1;
                }
            }
        }

        public void ResetLatents()
        {
            SampleLatents(out m_Latents, m_LatentSize);
        }

        public void ResetLatentStepCounts()
        {
            m_LatentStepCount = Random.Range(LatentStepsMin, LatentStepsMax);
        }

        void SampleLatents(out float[] latents, int n)
        {
            latents = new float[n];
            float denominator = 0f;

            for (int i = 0; i < n; i++)
            {
                latents[i] = (float)m_RandomNormal.NextDouble();
                denominator += Mathf.Pow(Mathf.Abs(latents[i]), 2);
            }

            denominator = Mathf.Pow(denominator, 0.5f);

            for (int i = 0; i < n; i++)
            {
                latents[i] /= denominator;
            }
        }
    }
}

