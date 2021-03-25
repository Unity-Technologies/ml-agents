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
        public bool done;
        public IReadOnlyList<TensorProxy> nextState;
    }

    internal class ReplayBuffer
    {
        List<Transition> m_Buffer;
        int m_CurrentIndex;
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
                m_Buffer.Add(new Transition() {
                    state = CopyTensorList(state),
                    action = info.storedActions,
                    reward = info.reward,
                    done = info.done,
                    nextState = CopyTensorList(nextState)
                });
            }
            else
            {
                m_Buffer[m_CurrentIndex] = new Transition() {
                    state = CopyTensorList(state),
                    action = info.storedActions,
                    reward = info.reward,
                    done = info.done,
                    nextState = CopyTensorList(nextState)
                };
            }
            m_CurrentIndex += 1;
            m_CurrentIndex = m_CurrentIndex % m_MaxSize;
        }

        public List<Transition> SampleBatch(int batchSize)
        {
            var indexList = SampleIndex(batchSize);
            var samples = new List<Transition>(batchSize);
            for (var i = 0; i < batchSize; i++)
            {
                samples.Add(m_Buffer[indexList[i]]);
            }
            return samples;
        }

        public List<Transition> SampleDummyBatch(int batchSize)
        {
            var samples = new List<Transition>(batchSize);
            for (var i = 0; i < batchSize; i++)
            {
                samples.Add(m_Buffer[0]);
            }
            return samples;
        }

        private List<int> SampleIndex(int batchSize)
        {
            if (batchSize > m_Buffer.Count * 2)
            {
                return new int[batchSize].ToList();
            }
            Random random = new Random();
            HashSet<int> index = new HashSet<int>();

            while (index.Count < batchSize)
            {
                index.Add(random.Next(m_Buffer.Count));
            }
            return index.ToList();
        }

        IReadOnlyList<TensorProxy> CopyTensorList(IReadOnlyList<TensorProxy> inputList)
        {
            var newList = new List<TensorProxy>();
            for (var i = 0; i < inputList.Count; i++)
            {
                newList.Add(TensorUtils.DeepCopy(inputList[i]));
            }
            return (IReadOnlyList<TensorProxy>) newList;
        }
    }
}
