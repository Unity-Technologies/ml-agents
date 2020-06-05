using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Inference.Utils;
using UnityEngine;

namespace Unity.MLAgents
{
    /// <summary>
    /// The types of distributions from which to sample reset parameters.
    /// </summary>
    internal enum SamplerType
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
    internal sealed class SamplerFactory
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        internal SamplerFactory()
        {
        }

        /// <summary>
        /// Create the sampling distribution described by the encoding.
        /// </summary>
        /// <param name="encoding"> List of floats the describe sampling destribution.</param>
        public Func<float> CreateSampler(IList<float> encoding, int seed)
        {
            if ((int)encoding[0] == (int)SamplerType.Uniform)
            {
                return CreateUniformSampler(encoding[1], encoding[2], seed);
            }
            else if ((int)encoding[0] == (int)SamplerType.Gaussian)
            {
                return CreateGaussianSampler(encoding[1], encoding[2], seed);
            }
            else{
                Debug.LogWarning("EnvironmentParametersChannel received an unknown data type.");
                return () => 0;
            }

        }

        public Func<float> CreateUniformSampler(float min, float max, int seed)
        {
            System.Random distr = new System.Random(seed);
            return () => min + (float)distr.NextDouble() * (max - min);
        }

        public Func<float> CreateGaussianSampler(float mean, float stddev, int seed)
        {
            RandomNormal distr = new RandomNormal(seed, mean, stddev);
            return () => (float)distr.NextDouble();
        }
    }
}
