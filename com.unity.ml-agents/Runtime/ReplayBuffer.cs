// Buffer for C# training

using System;
using System.Linq;
using Unity.Barracuda;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;

namespace Unity.MLAgents
{
    internal struct Transition
    {
        public List<Tensor> state;
        public ActionBuffers action;
        public float reward;
        public List<Tensor> nextState;
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

        public void Push(AgentInfo info, List<Tensor> state, List<Tensor> nextState)
        {
            m_Buffer[currentIndex] = new Transition() {state=state, action=info.storedActions, reward=info.reward, nextState=nextState};
            currentIndex += 1;
            currentIndex = currentIndex % m_MaxSize;
        }

        public Transition[] Sample(int batchSize)
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
