// Trainer for C# training. One trainer per behavior.

using System;
using Unity.MLAgents.Actuators;
using Unity.Barracuda;
using UnityEngine;

namespace Unity.MLAgents
{
    internal class TrainerConfig
    {
        public int bufferSize = 500000;
        public int batchSize = 100;
        public float gamma = 0.9f;
        public float learningRate = 0.0001f;
        public int updatePeriod = 500;
        public int numSamplingAndUpdates = 50;
        // public int updateTargetFreq = 200;
    }

    internal class Trainer : IDisposable
    {
        ReplayBuffer m_Buffer;
        TrainingModelRunner m_ModelRunner;
        TrainingModelRunner m_TargetModelRunner;
        string m_behaviorName;
        TrainerConfig m_Config;
        int m_TrainingStep;

        public Trainer(string behaviorName, ActionSpec actionSpec, NNModel model, int seed = 0, TrainerConfig config = null)
        {
            m_Config = config ?? new TrainerConfig();
            m_behaviorName = behaviorName;
            m_Buffer = new ReplayBuffer(m_Config.bufferSize);
            m_ModelRunner = new TrainingModelRunner(actionSpec, model, m_Buffer, m_Config, seed);
            // m_TargetModelRunner = new TrainingModelRunner(actionSpec, model, m_Buffer, m_Config, seed);
            // copy weights from model to target model
            // m_TargetModelRunner.model.weights = m_ModelRunner.model.weights
            Academy.Instance.TrainerUpdate += Update;
        }

        public string BehaviorName
        {
            get => m_behaviorName;
        }

        public ReplayBuffer Buffer
        {
            get => m_Buffer;
        }

        public TrainingModelRunner TrainerModelRunner
        {
            get => m_ModelRunner;
        }

        public void Dispose()
        {
            Academy.Instance.TrainerUpdate -= Update;
        }

        public void Update()
        {
            if (!MyTimeScaleSetting.instance.IsTraining)
            {
                return;
            }
            if (m_TrainingStep % m_Config.updatePeriod != 0)
            {
                m_TrainingStep += 1;
                return;
            }
            if (m_Buffer.Count < m_Config.batchSize * 2)
            {
                return;
            }

            float loss = 0f;
            for (var i = 0; i < m_Config.numSamplingAndUpdates; i++)
            {
                var samples = m_Buffer.SampleBatch(m_Config.batchSize);
                loss += m_ModelRunner.UpdateModel(samples);
            }
            Debug.Log($"Loss: {loss/m_Config.numSamplingAndUpdates}");
            m_ModelRunner.SaveModelToFile();

            // Update target network
            // if (m_TrainingStep % m_Config.updateTargetFreq == 0)
            // {
            //     // copy weights
            // }

            m_TrainingStep += 1;
        }
    }
}
