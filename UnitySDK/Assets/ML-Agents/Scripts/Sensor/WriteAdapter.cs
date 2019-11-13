using System.Collections.Generic;
using MLAgents.InferenceBrain;

namespace MLAgents.Sensor
{
    /// <summary>
    /// Allows sensors to write to both TensorProxy and float arrays/lists.
    /// </summary>
    public class WriteAdapter
    {
        IList<float> m_Data;
        int m_Offset;

        TensorProxy m_Proxy;
        int m_Batch;

        /// <summary>
        /// Set the adapter to write to an IList at the given channelOffset.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="offset"></param>
        public void SetTarget(IList<float> data, int offset)
        {
            m_Data = data;
            m_Offset = offset;
            m_Proxy = null;
            m_Batch = -1;
        }

        /// <summary>
        /// Set the adapter to write to a TensorProxy at the given batch and channel offset.
        /// </summary>
        /// <param name="tensorProxy"></param>
        /// <param name="batchIndex"></param>
        /// <param name="channelOffset"></param>
        public void SetTarget(TensorProxy tensorProxy, int batchIndex, int channelOffset)
        {
            m_Proxy = tensorProxy;
            m_Batch = batchIndex;
            m_Offset = channelOffset;
            m_Data = null;
        }

        /// <summary>
        /// 1D write access at a specified index. Use AddRange if possible instead.
        /// </summary>
        /// <param name="index">Index to write to</param>
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

        /// <summary>
        /// 3D write access at the specified height, width, and channel. Only usable with a TensorProxy target.
        /// </summary>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="ch"></param>
        public float this[int h, int w, int ch]
        {
            set
            {
                // Only TensorProxy supports 3D access
                m_Proxy.data[m_Batch, h, w, ch + m_Offset] = value;
            }
        }

        /// <summary>
        /// Write the range of floats
        /// </summary>
        /// <param name="data"></param>
        /// <param name="writeOffset">Optional write offset</param>
        public void AddRange(IEnumerable<float> data, int writeOffset = 0)
        {
            if (m_Data != null)
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Data[index + m_Offset + writeOffset] = val;
                    index++;
                }
            }
            else
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Proxy.data[m_Batch, index + m_Offset + writeOffset] = val;
                    index++;
                }
            }
        }
    }
}
