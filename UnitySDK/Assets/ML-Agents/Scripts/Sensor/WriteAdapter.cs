using System.Collections.Generic;
using MLAgents.InferenceBrain;

namespace MLAgents.Sensor
{
    public class WriteAdapter
    {
        IList<float> m_Data;
        int m_Offset;

        TensorProxy m_Proxy;
        int m_Batch;

        public WriteAdapter(){ }

        public void SetTarget(IList<float> data, int offset)
        {
            m_Data = data;
            m_Offset = offset;
            m_Proxy = null;
            m_Batch = -1;
        }

        public void SetTarget(TensorProxy tensorProxy, int batchIndex, int offset)
        {
            m_Proxy = tensorProxy;
            m_Batch = batchIndex;
            m_Offset = offset;
            m_Data = null;
        }

        public float this[int index]
        {
            set
            {
                if (m_Data != null)
                {
                    m_Data[index + m_Offset] = value;
                }
                else
                {
                    m_Proxy.data[m_Batch, index + m_Offset] = value;
                }
            }
        }

        public float this[int h, int w, int ch]
        {
            set
            {
                // Only TensorProxy supports 3D access
                m_Proxy.data[m_Batch, h, w, ch + m_Offset] = value;
            }
        }

        public void AddRange(IEnumerable<float> data, int offset = 0)
        {
            if (m_Data != null)
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Data[index + m_Offset + offset] = val;
                    index++;
                }
            }
            else
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Proxy.data[m_Batch, index + m_Offset + offset] = val;
                    index++;
                }
            }
        }
    }
}
