using System.Collections.Generic;
using MLAgents.InferenceBrain;
using UnityEngine;

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

        int[] m_Shape;

        /// <summary>
        /// Set the adapter to write to an IList at the given channelOffset.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="shape"></param>
        /// <param name="offset"></param>
        public void SetTarget(IList<float> data, int[] shape, int offset)
        {
            m_Data = data;
            m_Offset = offset;
            m_Proxy = null;
            m_Batch = 0;
            m_Shape = shape;
        }

        /// <summary>
        /// Set the adapter to write to a TensorProxy at the given batch and channel offset.
        /// </summary>
        /// <param name="tensorProxy"></param>
        /// <param name="shape"></param>
        /// <param name="batchIndex"></param>
        /// <param name="channelOffset"></param>
        public void SetTarget(TensorProxy tensorProxy, int[] shape, int batchIndex, int channelOffset)
        {
            m_Proxy = tensorProxy;
            m_Batch = batchIndex;
            m_Offset = channelOffset;
            m_Data = null;
            m_Shape = shape;
        }

        /// <summary>
        /// 1D write access at a specified index. Use AddRange if possible instead.
        /// </summary>
        /// <param name="index">Index to write to</param>
        public float this[int index]
        {
            set
            {
                // TODO check shape is 1D?
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
                if (m_Data != null)
                {
                    var height = m_Shape[0];
                    var width = m_Shape[1];
                    var channels = m_Shape[2];
                    Debug.Assert(h > 0 && h < height);
                    Debug.Assert(w >= 0 && w < width);
                    Debug.Assert(ch >= 0 && ch < channels);

                    // Math copied from TensorShape.Index(). Note that m_Batch should always be 0
                    var index = m_Batch * height * width * channels + h * width * channels + w * channels + ch;
                    m_Data[index + m_Offset] = value;
                }
                else
                {
                    m_Proxy.data[m_Batch, h, w, ch + m_Offset] = value;
                }
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
