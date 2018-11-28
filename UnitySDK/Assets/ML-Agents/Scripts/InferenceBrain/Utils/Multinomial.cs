using System;
using Assert = UnityEngine.Assertions.Assert;
using UnityEngine;

namespace MLAgents.InferenceBrain.Utils
{
    /// <summary>
    /// Multinomial - Draws samples from a multinomial distribution in log space
    /// Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/multinomial_op.cc
    /// </summary>
    public class Multinomial
    {
        private readonly System.Random m_random;

        public Multinomial(int seed)
        {
            m_random = new System.Random(seed);
        }

        /// <summary>
        /// Draw samples from a multinomial distribution based on log-probabilities specified in tensor src. The samples
        /// will be saved in the dst tensor.
        /// </summary>
        /// <param name="src">2-D tensor with shape batch_size x num_classes</param>
        /// <param name="dst">Allocated tensor with size batch_size x num_samples</param>
        /// <exception cref="NotImplementedException">Multinomial doesn't support integer tensors</exception>
        /// <exception cref="ArgumentException">Issue with tensor shape or type</exception>
        /// <exception cref="ArgumentNullException">At least one of the tensors is not allocated</exception>
        public void Eval(Tensor src, Tensor dst)
        {
            if (src.DataType != typeof(float))
            {
                throw new NotImplementedException("Multinomial does not support integer tensors yet!");
            }

            if (src.ValueType != dst.ValueType)
            {
                throw new ArgumentException("Source and destination tensors have different types!");
            }

            if (src.Data == null || dst.Data == null)
            {
                throw new ArgumentNullException();
            }

            float[,] input_data = src.Data as float[,];
            if (input_data == null)
            {
                throw new ArgumentException("Input data is not of the correct shape! Required batch x logits");
            }

            float[,] output_data = dst.Data as float[,];
            if (output_data == null)
            {
                throw new ArgumentException("Output data is not of the correct shape! Required batch x samples");
            }

            if (input_data.GetLength(0) != output_data.GetLength(0))
            {
                throw new ArgumentException("Batch size for input and output data is different!");
            }

            for (int batch = 0; batch < input_data.GetLength(0); ++batch)
            {
                // Find the class maximum
                float maxProb = float.NegativeInfinity;
                for (int cls = 0; cls < input_data.GetLength(1); ++cls)
                {
                    maxProb = Mathf.Max(input_data[batch, cls], maxProb);
                }
                
                // Sum the log probabilities and compute CDF
                float sumProb = 0.0f;
                float[] cdf = new float[input_data.GetLength(1)];
                for (int cls = 0; cls < input_data.GetLength(1); ++cls)
                {
                    sumProb += Mathf.Exp(input_data[batch, cls] - maxProb);
                    cdf[cls] = sumProb;
                }
                
                // Generate the samples
                for (int sample = 0; sample < output_data.GetLength(1); ++sample)
                {
                    float p = (float)m_random.NextDouble() * sumProb;
                    int cls = 0;
                    while (cdf[cls] < p)
                    {
                        ++cls;
                    }

                    output_data[batch, sample] = cls;
                }

            }
            
        }
    }
}
