namespace MLAgents.InferenceBrain.Utils
{
    /// <summary>
    /// Multinomial - Draws samples from a multinomial distribution given a (potentially unscaled)
    /// cumulative mass function (CMF). This means that the CMF need not "end" with probability
    /// mass of 1.0. For instance: [0.1, 0.2, 0.5] is a valid (unscaled). What is important is
    /// that it is a cumulative function, not a probability function. In other words,
    /// entry[i] = P(x \le i), NOT P(i - 1 \le x \lt i).
    /// (\le stands for less than or equal to while \lt is strictly less than).
    /// </summary>
    public class Multinomial
    {
        private readonly System.Random m_Random;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="seed">
        /// Seed for the random number generator used in the sampling process.
        /// </param>
        public Multinomial(int seed)
        {
            m_Random = new System.Random(seed);
        }

        /// <summary>
        /// Samples from the Multinomial distribution defined by the provided cumulative
        /// mass function.
        /// </summary>
        /// <param name="cmf">
        /// Cumulative mass function, which may be unscaled. The entries in this array need
        /// to be monotonic (always increasing). If the CMF is scaled, then the last entry in
        /// the array will be 1.0.
        /// </param>
        /// <returns>A sampled index from the CMF ranging from 0 to cmf.Length-1.</returns>
        public int Sample(float[] cmf)
        {
            var p = (float)m_Random.NextDouble() * cmf[cmf.Length - 1];
            var cls = 0;
            while (cmf[cls] < p)
            {
                ++cls;
            }

            return cls;
        }
    }
}
