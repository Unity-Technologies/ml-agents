namespace MLAgents.InferenceBrain.Utils
{
    /// <summary>
    /// Multinomial - Draws samples from a multinomial distribution given the CDF.
    /// </summary>
    public class Multinomial
    {
        private readonly System.Random _random;

        public Multinomial(int seed)
        {
            _random = new System.Random(seed);
        }

        public int Sample(float[] cdf)
        {
            var p = (float) _random.NextDouble() * cdf[cdf.Length - 1];
            var cls = 0;
            while (cdf[cls] < p)
            {
                ++cls;
            }

            return cls;
        }
    }
}
