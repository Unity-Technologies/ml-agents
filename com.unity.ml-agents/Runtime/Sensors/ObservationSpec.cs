namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A description of the observations that an ISensor produces.
    /// This includes the size of the observation, the properties of each dimension, and how the observation
    /// should be used for training.
    /// </summary>
    public struct ObservationSpec
    {
        internal readonly InplaceArray<int> m_Shape;

        /// <summary>
        /// The size of the observations that will be generated.
        /// For example, a sensor that observes the velocity of a rigid body (in 3D) would use [3].
        /// A sensor that returns an RGB image would use [Height, Width, 3].
        /// </summary>
        public InplaceArray<int> Shape
        {
            get => m_Shape;
        }

        internal readonly InplaceArray<DimensionProperty> m_DimensionProperties;

        /// <summary>
        /// The properties of each dimensions of the observation.
        /// The length of the array must be equal to the rank of the observation tensor.
        /// </summary>
        /// <remarks>
        /// It is generally recommended to use default values provided by helper functions,
        /// as not all combinations of DimensionProperty may be supported by the trainer.
        /// </remarks>
        public InplaceArray<DimensionProperty> DimensionProperties
        {
            get => m_DimensionProperties;
        }

        internal ObservationType m_ObservationType;

        /// <summary>
        /// The type of the observation, e.g. whether they are generic or
        /// help determine the goal for the Agent.
        /// </summary>
        public ObservationType ObservationType
        {
            get => m_ObservationType;
        }

        /// <summary>
        /// The number of dimensions of the observation.
        /// </summary>
        public int Rank
        {
            get { return Shape.Length; }
        }

        /// <summary>
        /// Construct an ObservationSpec for 1-D observations of the requested length.
        /// </summary>
        /// <param name="length"></param>
        /// <param name="obsType"></param>
        /// <returns></returns>
        public static ObservationSpec Vector(int length, ObservationType obsType = ObservationType.Default)
        {
            return new ObservationSpec(
                new InplaceArray<int>(length),
                new InplaceArray<DimensionProperty>(DimensionProperty.None),
                obsType
            );
        }

        /// <summary>
        /// Construct an ObservationSpec for variable-length observations.
        /// </summary>
        /// <param name="obsSize"></param>
        /// <param name="maxNumObs"></param>
        /// <returns></returns>
        public static ObservationSpec VariableLength(int obsSize, int maxNumObs)
        {
            var dimProps = new InplaceArray<DimensionProperty>(
                DimensionProperty.VariableSize,
                DimensionProperty.None
            );
            return new ObservationSpec(
                new InplaceArray<int>(obsSize, maxNumObs),
                dimProps
            );
        }

        /// <summary>
        /// Construct an ObservationSpec for visual-like observations, e.g. observations
        /// with a height, width, and possible multiple channels.
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="channels"></param>
        /// <param name="obsType"></param>
        /// <returns></returns>
        public static ObservationSpec Visual(int height, int width, int channels, ObservationType obsType = ObservationType.Default)
        {
            var dimProps = new InplaceArray<DimensionProperty>(
                DimensionProperty.TranslationalEquivariance,
                DimensionProperty.TranslationalEquivariance,
                DimensionProperty.None
            );
            return new ObservationSpec(
                new InplaceArray<int>(height, width, channels),
                dimProps,
                obsType
            );
        }

        /// <summary>
        /// Create a general ObservationSpec from the shape, dimension properties, and observation type.
        /// </summary>
        /// <remarks>
        /// Note that not all combinations of DimensionProperty may be supported by the trainer.
        /// shape and dimensionProperties must have the same size.
        /// </remarks>
        /// <param name="shape"></param>
        /// <param name="dimensionProperties"></param>
        /// <param name="observationType"></param>
        /// <exception cref="UnityAgentsException"></exception>
        public ObservationSpec(
            InplaceArray<int> shape,
            InplaceArray<DimensionProperty> dimensionProperties,
            ObservationType observationType = ObservationType.Default
        )
        {
            if (shape.Length != dimensionProperties.Length)
            {
                throw new UnityAgentsException("shape and dimensionProperties must have the same length.");
            }
            m_Shape = shape;
            m_DimensionProperties = dimensionProperties;
            m_ObservationType = observationType;
        }
    }
}
