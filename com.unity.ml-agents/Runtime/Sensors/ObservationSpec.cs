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

        public int Dimensions
        {
            get { return Shape.Length; }
        }

        // TODO RENAME?
        public static ObservationSpec FromShape(int length)
        {
            InplaceArray<int> shape = new InplaceArray<int>(length);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(DimensionProperty.None);
            return new ObservationSpec
            {
                ObservationType = ObservationType.Default,
                Shape = shape,
                DimensionProperties = dimProps
            };
        }

        public static ObservationSpec FromShape(int obsSize, int maxNumObs)
        {
            InplaceArray<int> shape = new InplaceArray<int>(obsSize, maxNumObs);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(DimensionProperty.VariableSize, DimensionProperty.None);
            return new ObservationSpec
            {
                ObservationType = ObservationType.Default,
                Shape = shape,
                DimensionProperties = dimProps
            };
        }

        public static ObservationSpec FromShape(int height, int width, int channels)
        {
            InplaceArray<int> shape = new InplaceArray<int>(height, width, channels);
            InplaceArray<DimensionProperty> dimProps = new InplaceArray<DimensionProperty>(
                DimensionProperty.TranslationalEquivariance, DimensionProperty.TranslationalEquivariance, DimensionProperty.None
            );
            return new ObservationSpec
            {
                ObservationType = ObservationType.Default,
                Shape = shape,
                DimensionProperties = dimProps
            };
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// Information about a single dimension. Future per-dimension properties can go here.
    /// This is nicer because it ensures the shape and dimension properties that the same size
    /// </summary>
    public struct DimensionInfo
    {
        public int Rank;
        public DimensionProperty DimensionProperty;
    }

    public struct ObservationSpecAlternativeOne
    {
        public ObservationType ObservationType;
        public DimensionInfo[] DimensionInfos;
        // Similar ObservationSpec.FromShape() as above
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// Uses Barracuda's TensorShape struct instead of an int[] for the shape.
    /// This doesn't fully avoid allocations because of DimensionProperty, so we'd need more supporting code.
    /// I don't like explicitly depending on Barracuda in one of our central interfaces, but listing as an alternative.
    /// </summary>
    public struct ObservationSpecAlternativeTwo
    {
        public ObservationType ObservationType;
        public TensorShape Shape;
        public DimensionProperty[] DimensionProperties;
    }
}
