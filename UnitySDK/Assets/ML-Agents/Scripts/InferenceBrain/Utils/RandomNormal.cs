using System;

namespace MLAgents.InferenceBrain.Utils
{
    /// <summary>
    /// RandomNormal - A random number generator that produces normally distributed random
    /// numbers using the Marsaglia polar method:
    /// https://en.wikipedia.org/wiki/Marsaglia_polar_method
    /// TODO: worth overriding System.Random instead of aggregating?
    /// </summary>
    public class RandomNormal
    {
        private readonly double m_Mean;
        private readonly double m_Stddev;
        private readonly Random m_Random;

        public RandomNormal(int seed, float mean = 0.0f, float stddev = 1.0f)
        {
            m_Mean = mean;
            m_Stddev = stddev;
            m_Random = new Random(seed);
        }

        // Each iteration produces two numbers. Hold one here for next call
        private bool m_HasSpare;
        private double m_SpareUnscaled;

        /// <summary>
        /// Return the next random double number
        /// </summary>
        /// <returns>Next random double number</returns>
        public double NextDouble()
        {
            if (m_HasSpare)
            {
                m_HasSpare = false;
                return m_SpareUnscaled * m_Stddev + m_Mean;
            }

            double u, v, s;
            do
            {
                u = m_Random.NextDouble() * 2.0 - 1.0;
                v = m_Random.NextDouble() * 2.0 - 1.0;
                s = u * u + v * v;
            }
            while (s >= 1.0 || Math.Abs(s) < double.Epsilon);

            s = Math.Sqrt(-2.0 * Math.Log(s) / s);
            m_SpareUnscaled = u * s;
            m_HasSpare = true;

            return v * s * m_Stddev + m_Mean;
        }
    }
}
