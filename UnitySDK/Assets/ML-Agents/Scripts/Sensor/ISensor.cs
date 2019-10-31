using System.Collections;
using System.Collections.Generic;
using MLAgents.InferenceBrain;

namespace MLAgents.Sensor
{
    public enum SensorCompressionType
    {
        None,
        PNG
    }

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

        public void AddRange(IEnumerable<float> data)
        {
            if (m_Data != null)
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Data[index + m_Offset] = val;
                    index++;
                }
            }
            else
            {
                int index = 0;
                foreach (var val in data)
                {
                    m_Proxy.data[m_Batch, index + m_Offset] = val;
                    index++;
                }
            }
        }
    }

    /// <summary>
    /// Sensor interface for generating observations.
    /// For custom implementations, it is recommended to SensorBase instead.
    /// </summary>
    public interface ISensor {
        /// <summary>
        /// Returns the size of the observations that will be generated.
        /// For example, a sensor that observes the velocity of a rigid body (in 3D) would return new {3}.
        /// A sensor that returns an RGB image would return new [] {Width, Height, 3}
        /// </summary>
        /// <returns></returns>
        int[] GetFloatObservationShape();

        /// <summary>
        /// Write the observation data directly to the WriteAdapter.
        /// This is considered an advanced interface; for a simpler approach, use SensorBase and override WriteFloats instead.
        /// </summary>
        /// <param name="adapater"></param>
        /// <returns>The number of elements written</returns>
        int Write(WriteAdapter adapater);

        /// <summary>
        /// Return a compressed representation of the observation. For small observations, this should generally not be
        /// implemented. However, compressing large observations (such as visual results) can significantly improve
        /// model training time.
        /// </summary>
        /// <returns></returns>
        byte[] GetCompressedObservation();

        /// <summary>
        /// Return the compression type being used. If no compression is used, return SensorCompressionType.None
        /// </summary>
        /// <returns></returns>
        SensorCompressionType GetCompressionType();

        /// <summary>
        /// Get the name of the sensor. This is used to ensure deterministic sorting of the sensors on an Agent,
        /// so the naming must be consistent across all sensors and agents.
        /// </summary>
        /// <returns>The name of the sensor</returns>
        string GetName();
    }

}
