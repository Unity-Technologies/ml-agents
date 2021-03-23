// Buffer for C# training

using System;
using System.Linq;
using Unity.Barracuda;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;

namespace Unity.MLAgents
{
    internal struct Transition
    {
        public IReadOnlyList<TensorProxy> state;
        public ActionBuffers action;
        public float reward;
        public IReadOnlyList<TensorProxy> nextState;
    }

    internal class ReplayBuffer
    {
        List<Transition> m_Buffer;
        int currentIndex;
        int m_MaxSize;

        public ReplayBuffer(int maxSize)
        {
            m_Buffer = new List<Transition>();
            m_Buffer.Capacity = maxSize;
            m_MaxSize = maxSize;
        }

        public int Count
        {
            get => m_Buffer.Count;
        }

        public void Push(AgentInfo info, IReadOnlyList<TensorProxy> state, IReadOnlyList<TensorProxy> nextState)
        {
            if (m_Buffer.Count < m_MaxSize)
            {
                m_Buffer.Append(new Transition() {state=state, action=info.storedActions, reward=info.reward, nextState=nextState});
            }
            else
            {
                m_Buffer[currentIndex] = new Transition() {state=state, action=info.storedActions, reward=info.reward, nextState=nextState};
            }
            currentIndex += 1;
            currentIndex = currentIndex % m_MaxSize;
        }

        public Transition[] SampleBatch(int batchSize)
        {
            var indexList = SampleIndex(batchSize);
            var samples = new Transition[batchSize];
            for (var i = 0; i < batchSize; i++)
            {
                samples[i] = m_Buffer[indexList[i]];
            }
            return samples;
        }

        private List<int> SampleIndex(int batchSize)
        {
            Random random = new Random();
            HashSet<int> index = new HashSet<int>();

            while (index.Count < batchSize)
            {
                index.Add(random.Next(m_Buffer.Count));
            }
            return index.ToList();
        }
    }
}
