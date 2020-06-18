using Unity.MLAgents.Sensors;

namespace Examples.GridWorld.Scripts
{
    public class GridSensorComponent : SensorComponent
    {
        public GridArea gridArea;
        int pixelsPerCell = 8;
        // TODO use grid size from env parameters
        int gridSize = 5;

        /// <summary>
        /// Creates a BasicSensor.
        /// </summary>
        /// <returns></returns>
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
        int m_PixlesPerCell;
        int m_GridSize;
        int[] m_Shape;
        const int k_NumChannels = 4;

        public GridSensor(GridArea gridArea, int gridSize, int pixelsPerCell)
        {
            m_GridArea = gridArea;
            m_GridSize = gridSize;
            m_PixlesPerCell = pixelsPerCell;

            m_Shape = new []{ gridSize * pixelsPerCell, gridSize * pixelsPerCell, k_NumChannels };
        }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            var board = m_GridArea.board;
            var height = m_GridSize * m_PixlesPerCell;
            var width = m_GridSize * m_PixlesPerCell;
            for (var h = 0; h < height; h++)
            {
                var i = h / m_PixlesPerCell;
                for (var w = 0; w <  width; w++)
                {
                    var j = w / m_PixlesPerCell;
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
