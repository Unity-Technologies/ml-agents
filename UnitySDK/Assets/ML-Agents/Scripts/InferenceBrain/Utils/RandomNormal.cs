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
        private readonly double _mean;
        private readonly double _stddev;
        private readonly Random _random;

        public RandomNormal(int seed, float mean = 0.0f, float stddev = 1.0f)
        {
            _mean = mean;
            _stddev = stddev;
            _random = new Random(seed);
        }

        // Each iteration produces two numbers. Hold one here for next call
        private bool _hasSpare;
        private double _spareUnscaled;

        /// <summary>
        /// Return the next random double number
        /// </summary>
        /// <returns>Next random double number</returns>
        public double NextDouble()
        {
            if (_hasSpare)
            {
                _hasSpare = false;
                return _spareUnscaled * _stddev + _mean;
            }

            double u, v, s;
            do
            {
                u = _random.NextDouble() * 2.0 - 1.0;
                v = _random.NextDouble() * 2.0 - 1.0;
                s = u * u + v * v;
            }
            while (s >= 1.0 || Math.Abs(s) < double.Epsilon);

            s = Math.Sqrt(-2.0 * Math.Log(s) / s);
            _spareUnscaled = u * s;
            _hasSpare = true;

            return v * s * _stddev + _mean;
        }
    }
}
