namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The Dimension property flags of the observations
    /// </summary>
    [System.Flags]
    public enum DimensionProperty
    {
        /// <summary>
        /// No properties specified.
        /// </summary>
        Unspecified = 0,

        /// <summary>
        /// No Property of the observation in that dimension. Observation can be processed with
        /// fully connected networks.
        /// </summary>
        None = 1,

        /// <summary>
        /// Means it is suitable to do a convolution in this dimension.
        /// </summary>
        TranslationalEquivariance = 2,

        /// <summary>
        /// Means that there can be a variable number of observations in this dimension.
        /// The observations are unordered.
        /// </summary>
        VariableSize = 4,
    }


    /// <summary>
    /// Sensor interface for sensors with special dimension properties.
    /// </summary>
    internal interface IDimensionPropertiesSensor
    {
        /// <summary>
        /// Returns the array containing the properties of each dimensions of the
        /// observation. The length of the array must be equal to the rank of the
        /// observation tensor.
        /// </summary>
        /// <returns>The array of DimensionProperty</returns>
        DimensionProperty[] GetDimensionProperties();
    }
}
