namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// The compression setting for visual/camera observations.
    /// </summary>
    public enum SensorCompressionType
    {
        /// <summary>
        /// No compression. Data is preserved as float arrays.
        /// </summary>
        None,

        /// <summary>
        /// PNG format. Data will be stored in binary format.
        /// </summary>
        PNG
    }


    /// <summary>
    /// The ObservationType enum of the Sensor.
    /// </summary>
    public enum ObservationType
    {
        // Collected observations are generic.
        Default = 0,
        // Collected observations contain goal information.
        Goal = 1,
        // Collected observations contain reward information.
        Reward = 2,
        // Collected observations are messages from other agents.
        Message = 3,
    }

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
    /// Sensor interface for generating observations.
    /// </summary>
    public interface ISensor
    {
        /// <summary>
        /// Returns the size of the observations that will be generated.
        /// For example, a sensor that observes the velocity of a rigid body (in 3D) would return
        /// new {3}. A sensor that returns an RGB image would return new [] {Height, Width, 3}
        /// </summary>
        /// <returns>Size of the observations that will be generated.</returns>
        int[] GetObservationShape();

        /// <summary>
        /// Write the observation data directly to the <see cref="ObservationWriter"/>.
        /// Note that this (and  <see cref="GetCompressedObservation"/>) may
        /// be called multiple times per agent step, so should not mutate any internal state.
        /// </summary>
        /// <param name="writer">Where the observations will be written to.</param>
        /// <returns>The number of elements written.</returns>
        int Write(ObservationWriter writer);

        /// <summary>
        /// Return a compressed representation of the observation. For small observations,
        /// this should generally not be implemented. However, compressing large observations
        /// (such as visual results) can significantly improve model training time.
        /// </summary>
        /// <returns>Compressed observation.</returns>
        byte[] GetCompressedObservation();

        /// <summary>
        /// Update any internal state of the sensor. This is called once per each agent step.
        /// </summary>
        void Update();

        /// <summary>
        /// Resets the internal state of the sensor. This is called at the end of an Agent's episode.
        /// Most implementations can leave this empty.
        /// </summary>
        void Reset();

        /// <summary>
        /// Return the compression type being used. If no compression is used, return
        /// <see cref="SensorCompressionType.None"/>.
        /// </summary>
        /// <returns>Compression type used by the sensor.</returns>
        SensorCompressionType GetCompressionType();

        /// <summary>
        /// Get the name of the sensor. This is used to ensure deterministic sorting of the sensors
        /// on an Agent, so the naming must be consistent across all sensors and agents.
        /// </summary>
        /// <returns>The name of the sensor.</returns>
        string GetName();

        /// <summary>
        /// Returns the ObservationType enum corresponding to the type of the sensor.
        /// Most ISensor implementations should return ObservationType.Default.
        /// </summary>
        /// <returns>The ObservationType enum</returns>
        ObservationType GetObservationType();

        /// <summary>
        /// Returns the array containing the properties of each dimensions of the
        /// observation. The length of the array must be equal to the length of the
        /// array returned by GetObservationShape().
        /// </summary>
        /// <returns>The array of DimensionProperty</returns>
        DimensionProperty[] GetDimensionProperties();
    }


    /// <summary>
    /// Helper methods to be shared by all classes that implement <see cref="ISensor"/>.
    /// </summary>
    public static class SensorExtensions
    {
        /// <summary>
        /// Get the total number of elements in the ISensor's observation (i.e. the product of the
        /// shape elements).
        /// </summary>
        /// <param name="sensor"></param>
        /// <returns></returns>
        public static int ObservationSize(this ISensor sensor)
        {
            var shape = sensor.GetObservationShape();
            var count = 1;
            foreach (var dim in shape)
            {
                count *= dim;
            }

            return count;
        }

        /// <summary>
        /// Get a default DimensionProperty for the sensor. Note that this allocates a new array,
        /// so the results should be cached in the Sensor implementation.
        /// </summary>
        /// <param name="sensor"></param>
        /// <returns></returns>
        public static DimensionProperty[] DefaultDimensionProperties(this ISensor sensor)
        {
            var shape = sensor.GetObservationShape();
            if (shape.Length == 3)
            {
                return new[] {
                    DimensionProperty.TranslationalEquivariance,
                    DimensionProperty.TranslationalEquivariance,
                    DimensionProperty.None
                };
            }

            var result = new DimensionProperty[shape.Length];
            for (var i = 0; i < shape.Length; i++)
            {
                result[i] = DimensionProperty.Unspecified;
            }

            return result;
        }
    }
}
