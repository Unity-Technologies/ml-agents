using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Inference.Utils;
using UnityEngine;
using Random=UnityEngine.Random;

namespace Unity.MLAgents
{
    /// <summary>
    /// The types of distributions from which to sample reset parameters.
    /// </summary>
    public enum SamplerType
    {
        /// <summary>
        /// Samples a reset parameter from a uniform distribution.
        /// </summary>
        Uniform = 0,

        /// <summary>
        /// Samples a reset parameter from a Gaussian distribution.
        /// </summary>
        Gaussian = 1
    }

    /// <summary>
    /// Takes a list of floats that encode a sampling distribution and returns the sampling function.
    /// </summary>
    public sealed class SamplerFactory
    {

        int m_Seed;

        /// <summary>
        /// Constructor.
        /// </summary>
        internal SamplerFactory(int seed)
        {
            m_Seed = seed;    
        }

        /// <summary>
        /// Create the sampling distribution described by the encoding.
        /// </summary>
        /// <param name="encoding"> List of floats the describe sampling destribution.</param>
        public Func<float> CreateSampler(IList<float> encoding)
        {
            if ((int)encoding[0] == (int)SamplerType.Uniform)
            {
                return CreateUniformSampler(encoding[1], encoding[2]);
            }
            else if ((int)encoding[0] == (int)SamplerType.Gaussian)
            {
                return CreateGaussianSampler(encoding[1], encoding[2]);
            }
            else{
                Debug.LogWarning("EnvironmentParametersChannel received an unknown data type.");
                return () => 0;
            }

        }

        public Func<float> CreateUniformSampler(float min, float max)
        {
            return () => Random.Range(min, max);
        }

        public Func<float> CreateGaussianSampler(float mean, float stddev)
        {
            RandomNormal distr = new RandomNormal(m_Seed, mean, stddev);
            return () => (float)distr.NextDouble();
        }
    }
}
