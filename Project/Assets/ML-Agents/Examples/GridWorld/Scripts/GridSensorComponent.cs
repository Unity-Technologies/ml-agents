using Unity.MLAgents.Sensors;

namespace Examples.GridWorld.Scripts
{
    public class GridSensorComponent : SensorComponent
    {
        public GridArea gridArea;
        int pixelsPerCell = 8;
        // TODO use grid size from env parameters
        int gridSize = 5;

        public override ISensor CreateSensor()
        {
            return new GridSensor(gridArea, gridSize, pixelsPerCell);
        }

        /// <inheritdoc/>
        public override int[] GetObservationShape()
        {
            return new[] { gridSize * pixelsPerCell, gridSize * pixelsPerCell, 4 };
        }
    }

    public class GridSensor : ISensor
    {
        GridArea m_GridArea;
        int m_PixelsPerCell;
        int m_GridSize;
        int[] m_Shape;
        const int k_NumChannels = 4;

        public GridSensor(GridArea gridArea, int gridSize, int pixelsPerCell)
        {
            m_GridArea = gridArea;
            m_GridSize = gridSize;
            m_PixelsPerCell = pixelsPerCell;

            m_Shape = new []{ gridSize * pixelsPerCell, gridSize * pixelsPerCell, k_NumChannels };
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <summary>
        /// Writes a one-hot encoding of the area state for the observations.
        /// As a 3x3 example, for this area state:
        ///
        ///  A..
        ///  ..P
        ///  G..
        ///
        /// The corresponding channels would be
        /// channel 0 (empty)
        ///   011
        ///   110
        ///   011
        ///
        /// channel 1 (goal)
        ///   000
        ///   000
        ///   100
        ///
        /// channel 2 (pit)
        ///   000
        ///   001
        ///   000
        ///
        /// channel 3 (agent)
        ///   100
        ///   000
        ///   000
        ///
        /// </summary>
        /// <param name="writer"></param>
        /// <returns></returns>
        public int Write(ObservationWriter writer)
        {
            // There is a minimum size to visual observations (see MIN_RESOLUTION_FOR_ENCODER in the python code)
            // So repeat each cell m_PixelsPerCell times/
            var board = m_GridArea.board;
            var height = m_GridSize * m_PixelsPerCell;
            var width = m_GridSize * m_PixelsPerCell;
            for (var h = 0; h < height; h++)
            {
                var i = h / m_PixelsPerCell;
                for (var w = 0; w <  width; w++)
                {
                    var j = w / m_PixelsPerCell;
                    var cellVal = board[i, j];
                    for (var c = 0; c < k_NumChannels; c++)
                    {
                        writer[h, w, c] = (c == (int)cellVal) ? 1.0f : 0.0f;
                    }
                }
            }

            var numWritten = height * width * k_NumChannels;
            return numWritten;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public void Update() { }

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return "GridSensor";
        }
    }
}
