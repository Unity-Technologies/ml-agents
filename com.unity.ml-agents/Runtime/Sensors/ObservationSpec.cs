using Unity.Barracuda;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// This is the simplest approach, but there's possible user error if Shape.Length != DimensionProperties.Length
    /// </summary>
    public struct ObservationSpec
    {
        public ObservationType ObservationType;
        public int[] Shape;
        public DimensionProperty[] DimensionProperties;

        /// <summary>
        /// Create an Observation spec with default DimensionProperties and ObservationType from the shape.
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static ObservationSpec FromShape(params int[] shape)
        {
            DimensionProperty[] dimProps = null;
            if (shape.Length == 1)
            {
                dimProps = new[] { DimensionProperty.None };
            }
            else if (shape.Length == 2)
            {
                // NOTE: not sure if I like this - might leave Unspecified and make BufferSensor set it
                dimProps = new[] { DimensionProperty.VariableSize, DimensionProperty.None };
            }
            else if (shape.Length == 3)
            {
                dimProps = new[]
                {
                    DimensionProperty.TranslationalEquivariance,
                    DimensionProperty.TranslationalEquivariance,
                    DimensionProperty.None
                };
            }
            else
            {
                dimProps = new DimensionProperty[shape.Length];
                for (var i = 0; i < dimProps.Length; i++)
                {
                    dimProps[i] = DimensionProperty.Unspecified;
                }
            }

            return new ObservationSpec
            {
                ObservationType = ObservationType.Default,
                Shape = shape,
                DimensionProperties = dimProps
            };
        }

        public ObservationSpec Clone()
        {
            return new ObservationSpec
            {
                Shape = (int[])Shape.Clone(),
                DimensionProperties = (DimensionProperty[])DimensionProperties.Clone(),
                ObservationType = ObservationType
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
