// Trainer for C# training. One trainer per behavior.

using System;
using Unity.MLAgents.Actuators;
using Unity.Barracuda;


namespace Unity.MLAgents
{
    internal class Trainer: IDisposable
    {
        ReplayBuffer m_Buffer;
        TrainingModelRunner m_ModelRunner;
        string m_behaviorName;
        int m_BufferSize = 1024;
        int batchSize = 64;
        float GAMMA;

        public Trainer(string behaviorName, ActionSpec actionSpec, NNModel model, int seed=0)
        {
            m_behaviorName = behaviorName;
            m_Buffer = new ReplayBuffer(m_BufferSize);
            m_ModelRunner = new TrainingModelRunner(actionSpec, model, seed);
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
            if (m_Buffer.Count < batchSize * 2)
            {
                return;
            }

            var samples = m_Buffer.SampleBatch(batchSize);
            // states = [s.state for s in samples]
            // actions = [s.action for s in samples]
            // q_values = policy_net(states).gather(1, action_batch)

            // next_states = [s.next_state for s in samples]
            // rewards = [s.reward for s in samples]
            // next_state_values = target_net(non_final_next_states).max(1)[0]
            // expected_q_values = (next_state_values * GAMMA) + rewards

            // loss = MSE(q_values, expected_q_values);
            // m_ModelRunner.model = Barracuda.ModelUpdate(m_ModelRunner.model, loss);
        }
    }
}
