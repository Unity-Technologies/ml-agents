using Unity.Barracuda;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// This is the simplest approach, but there's possible user error if Shape.Length != DimensionProperties.Length
    /// </summary>
    public struct ObservationSpec
    {
        public ObservationType ObservationType;
        public InplaceArray<int> Shape;
        public InplaceArray<DimensionProperty> DimensionProperties;

        public int NumDimensions
        {
            get { return Shape.Length; }
        }

        public static ObservationSpec Vector(int length)
        {
            InplaceArray<int> shape = new InplaceArray<int>(length);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(DimensionProperty.None);
            return new ObservationSpec(shape, dimProps);
        }

        public static ObservationSpec VariableSize(int obsSize, int maxNumObs)
        {
            InplaceArray<int> shape = new InplaceArray<int>(obsSize, maxNumObs);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(DimensionProperty.VariableSize, DimensionProperty.None);
            return new ObservationSpec(shape, dimProps);
        }

        public static ObservationSpec Visual(int height, int width, int channels)
        {
            InplaceArray<int> shape = new InplaceArray<int>(height, width, channels);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(
                DimensionProperty.TranslationalEquivariance, DimensionProperty.TranslationalEquivariance, DimensionProperty.None
            );
            return new ObservationSpec(shape, dimProps);
        }

        internal ObservationSpec(
            InplaceArray<int> shape,
            InplaceArray<DimensionProperty> dimensionProperties,
            ObservationType observationType = ObservationType.Default
        )
        {
            if (shape.Length != dimensionProperties.Length)
            {
                throw new UnityAgentsException("shape and dimensionProperties must have the same length.");
            }
            Shape = shape;
            DimensionProperties = dimensionProperties;
            ObservationType = observationType;
        }
    }

}
