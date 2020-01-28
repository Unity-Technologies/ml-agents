using System;
using System.Collections.Generic;
using Barracuda;
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

        TensorShape m_TensorShape;

        /// <summary>
        /// Set the adapter to write to an IList at the given channelOffset.
        /// </summary>
        /// <param name="data">Float array or list that will be written to.</param>
        /// <param name="shape">Shape of the observations to be written.</param>
        /// <param name="offset">Offset from the start of the float data to write to.</param>
        public void SetTarget(IList<float> data, int[] shape, int offset)
        {
            m_Data = data;
            m_Offset = offset;
            m_Proxy = null;
            m_Batch = 0;

            if (shape.Length == 1)
            {
                m_TensorShape = new TensorShape(m_Batch, shape[0]);
            }
            else
            {
                m_TensorShape = new TensorShape(m_Batch, shape[0], shape[1], shape[2]);
            }
        }

        /// <summary>
        /// Set the adapter to write to a TensorProxy at the given batch and channel offset.
        /// </summary>
        /// <param name="tensorProxy">Tensor proxy that will be writtent to.</param>
        /// <param name="batchIndex">Batch index in the tensor proxy (i.e. the index of the Agent)</param>
        /// <param name="channelOffset">Offset from the start of the channel to write to.</param>
        public void SetTarget(TensorProxy tensorProxy, int batchIndex, int channelOffset)
        {
            m_Proxy = tensorProxy;
            m_Batch = batchIndex;
            m_Offset = channelOffset;
            m_Data = null;
            m_TensorShape = m_Proxy.data.shape;
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
                if (m_Data != null)
                {
                    if (h < 0 || h >= m_TensorShape.height)
                    {
                        throw new IndexOutOfRangeException($"height value {h} must be in range [0, {m_TensorShape.height - 1}]");
                    }
                    if (w < 0 || w >= m_TensorShape.width)
                    {
                        throw new IndexOutOfRangeException($"width value {w} must be in range [0, {m_TensorShape.width - 1}]");
                    }
                    if (ch < 0 || ch >= m_TensorShape.channels)
                    {
                        throw new IndexOutOfRangeException($"channel value {ch} must be in range [0, {m_TensorShape.channels - 1}]");
                    }

                    var index = m_TensorShape.Index(m_Batch, h, w, ch + m_Offset);
                    m_Data[index] = value;
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
