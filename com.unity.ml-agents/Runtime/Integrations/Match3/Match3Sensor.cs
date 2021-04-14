using System;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Integrations.Match3
{
    /// <summary>
    /// Delegate that provides integer values at a given (x,y) coordinate.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    public delegate int GridValueProvider(int x, int y);

    /// <summary>
    /// Type of observations to generate.
    ///
    /// </summary>
    public enum Match3ObservationType
    {
        /// <summary>
        /// Generate a one-hot encoding of the cell type for each cell on the board. If there are special types,
        /// these will also be one-hot encoded.
        /// </summary>
        Vector,

        /// <summary>
        /// Generate a one-hot encoding of the cell type for each cell on the board, but arranged as
        /// a Rows x Columns visual observation. If there are special types, these will also be one-hot encoded.
        /// </summary>
        UncompressedVisual,

        /// <summary>
        /// Generate a one-hot encoding of the cell type for each cell on the board, but arranged as
        /// a Rows x Columns visual observation. If there are special types, these will also be one-hot encoded.
        /// During training, these will be sent as a concatenated series of PNG images, with 3 channels per image.
        /// </summary>
        CompressedVisual
    }

    /// <summary>
    /// Sensor for Match3 games. Can generate either vector, compressed visual,
    /// or uncompressed visual observations. Uses a GridValueProvider to determine the observation values.
    /// </summary>
    public class Match3Sensor : ISensor, IBuiltInSensor, IDisposable
    {
        Match3ObservationType m_ObservationType;
        ObservationSpec m_ObservationSpec;
        string m_Name;

        AbstractBoard m_Board;
        BoardSize m_MaxBoardSize;
        GridValueProvider m_GridValues;
        int m_OneHotSize;

        Texture2D m_ObservationTexture;
        OneHotToTextureUtil m_TextureUtil;

        /// <summary>
        /// Create a sensor for the GridValueProvider with the specified observation type.
        /// </summary>
        /// <remarks>
        /// Use Match3Sensor.CellTypeSensor() or Match3Sensor.SpecialTypeSensor() instead of calling
        /// the constructor directly.
        /// </remarks>
        /// <param name="board">The abstract board.</param>
        /// <param name="gvp">The GridValueProvider, should be either board.GetCellType or board.GetSpecialType.</param>
        /// <param name="oneHotSize">The number of possible values that the GridValueProvider can return.</param>
        /// <param name="obsType">Whether to produce vector or visual observations</param>
        /// <param name="name">Name of the sensor.</param>
        public Match3Sensor(AbstractBoard board, GridValueProvider gvp, int oneHotSize, Match3ObservationType obsType, string name)
        {
            var maxBoardSize = board.GetMaxBoardSize();
            m_Name = name;
            m_MaxBoardSize = maxBoardSize;
            m_GridValues = gvp;
            m_OneHotSize = oneHotSize;
            m_Board = board;

            m_ObservationType = obsType;
            m_ObservationSpec = obsType == Match3ObservationType.Vector
                ? ObservationSpec.Vector(maxBoardSize.Rows * maxBoardSize.Columns * oneHotSize)
                : ObservationSpec.Visual(maxBoardSize.Rows, maxBoardSize.Columns, oneHotSize);
        }

        /// <summary>
        /// Create a sensor that encodes the board cells as observations.
        /// </summary>
        /// <param name="board">The abstract board.</param>
        /// <param name="obsType">Whether to produce vector or visual observations</param>
        /// <param name="name">Name of the sensor.</param>
        /// <returns></returns>
        public static Match3Sensor CellTypeSensor(AbstractBoard board, Match3ObservationType obsType, string name)
        {
            var maxBoardSize = board.GetMaxBoardSize();
            return new Match3Sensor(board, board.GetCellType, maxBoardSize.NumCellTypes, obsType, name);
        }

        /// <summary>
        /// Create a sensor that encodes the cell special types as observations. Returns null if the board's
        /// NumSpecialTypes is 0 (indicating the sensor isn't needed).
        /// </summary>
        /// <param name="board">The abstract board.</param>
        /// <param name="obsType">Whether to produce vector or visual observations</param>
        /// <param name="name">Name of the sensor.</param>
        /// <returns></returns>
        public static Match3Sensor SpecialTypeSensor(AbstractBoard board, Match3ObservationType obsType, string name)
        {
            var maxBoardSize = board.GetMaxBoardSize();
            if (maxBoardSize.NumSpecialTypes == 0)
            {
                return null;
            }
            var specialSize = maxBoardSize.NumSpecialTypes + 1;
            return new Match3Sensor(board, board.GetSpecialType, specialSize, obsType, name);
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            m_Board.CheckBoardSizes(m_MaxBoardSize);
            var currentBoardSize = m_Board.GetCurrentBoardSize();

            int offset = 0;
            var isVisual = m_ObservationType != Match3ObservationType.Vector;

            // This is equivalent to
            // for (var r = 0; r < m_MaxBoardSize.Rows; r++)
            //     for (var c = 0; c < m_MaxBoardSize.Columns; c++)
            //          if (r < currentBoardSize.Rows && c < currentBoardSize.Columns)
            //              WriteOneHot
            //          else
            //              WriteZero
            // but rearranged to avoid the branching.

            for (var r = 0; r < currentBoardSize.Rows; r++)
            {
                for (var c = 0; c < currentBoardSize.Columns; c++)
                {
                    var val = m_GridValues(r, c);
                    writer.WriteOneHot(offset, r, c, val, m_OneHotSize, isVisual);
                    offset += m_OneHotSize;
                }

                for (var c = currentBoardSize.Columns; c < m_MaxBoardSize.Columns; c++)
                {
                    writer.WriteZero(offset, r, c, m_OneHotSize, isVisual);
                    offset += m_OneHotSize;
                }
            }

            for (var r = currentBoardSize.Rows; r < m_MaxBoardSize.Columns; r++)
            {
                for (var c = 0; c < m_MaxBoardSize.Columns; c++)
                {
                    writer.WriteZero(offset, r, c, m_OneHotSize, isVisual);
                    offset += m_OneHotSize;
                }
            }

            return offset;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            m_Board.CheckBoardSizes(m_MaxBoardSize);
            var height = m_MaxBoardSize.Rows;
            var width = m_MaxBoardSize.Columns;
            if (ReferenceEquals(null, m_ObservationTexture))
            {
                m_ObservationTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            }

            if (ReferenceEquals(null, m_TextureUtil))
            {
                m_TextureUtil = new OneHotToTextureUtil(height, width);
            }
            var bytesOut = new List<byte>();
            var currentBoardSize = m_Board.GetCurrentBoardSize();

            // Encode the cell types or special types as batches of PNGs
            // This is potentially wasteful, e.g. if there are 4 cell types and 1 special type, we could
            // fit in in 2 images, but we'll use 3 total (2 PNGs for the 4 cell type channels, and 1 for
            // the special types).
            var numCellImages = (m_OneHotSize + 2) / 3;
            for (var i = 0; i < numCellImages; i++)
            {
                m_TextureUtil.EncodeToTexture(
                    m_GridValues,
                    m_ObservationTexture,
                    3 * i,
                    currentBoardSize.Rows,
                    currentBoardSize.Columns
                );
                bytesOut.AddRange(m_ObservationTexture.EncodeToPNG());
            }

            return bytesOut.ToArray();
        }

        /// <inheritdoc/>
        public void Update()
        {
        }

        /// <inheritdoc/>
        public void Reset()
        {
        }

        internal SensorCompressionType GetCompressionType()
        {
            return m_ObservationType == Match3ObservationType.CompressedVisual ?
                SensorCompressionType.PNG :
                SensorCompressionType.None;
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return new CompressionSpec(GetCompressionType());
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.Match3Sensor;
        }

        /// <summary>
        /// Clean up the owned Texture2D.
        /// </summary>
        public void Dispose()
        {
            if (!ReferenceEquals(null, m_ObservationTexture))
            {
                Utilities.DestroyTexture(m_ObservationTexture);
                m_ObservationTexture = null;
            }
        }
    }

    /// <summary>
    /// Utility class for converting a 2D array of ints representing a one-hot encoding into
    /// a texture, suitable for conversion to PNGs for observations.
    /// Works by encoding 3 values at a time as pixels in the texture, thus it should be
    /// called (maxValue + 2) / 3 times, increasing the channelOffset by 3 each time.
    /// </summary>
    internal class OneHotToTextureUtil
    {
        Color[] m_Colors;
        int m_MaxHeight;
        int m_MaxWidth;
        private static Color[] s_OneHotColors = { Color.red, Color.green, Color.blue };

        public OneHotToTextureUtil(int maxHeight, int maxWidth)
        {
            m_Colors = new Color[maxHeight * maxWidth];
            m_MaxHeight = maxHeight;
            m_MaxWidth = maxWidth;
        }

        public void EncodeToTexture(
            GridValueProvider gridValueProvider,
            Texture2D texture,
            int channelOffset,
            int currentHeight,
            int currentWidth
        )
        {
            var i = 0;
            // There's an implicit flip converting to PNG from texture, so make sure we
            // counteract that when forming the texture by iterating through h in reverse.
            for (var h = m_MaxHeight - 1; h >= 0; h--)
            {
                for (var w = 0; w < m_MaxWidth; w++)
                {
                    var colorVal = Color.black;
                    if (h < currentHeight && w < currentWidth)
                    {
                        int oneHotValue = gridValueProvider(h, w);
                        if (oneHotValue >= channelOffset && oneHotValue < channelOffset + 3)
                        {
                            colorVal = s_OneHotColors[oneHotValue - channelOffset];
                        }
                    }
                    m_Colors[i++] = colorVal;
                }
            }
            texture.SetPixels(m_Colors);
        }
    }

    /// <summary>
    /// Utility methods for writing one-hot observations.
    /// </summary>
    internal static class ObservationWriterMatch3Extensions
    {
        public static void WriteOneHot(this ObservationWriter writer, int offset, int row, int col, int value, int oneHotSize, bool isVisual)
        {
            if (isVisual)
            {
                for (var i = 0; i < oneHotSize; i++)
                {
                    writer[row, col, i] = (i == value) ? 1.0f : 0.0f;
                }
            }
            else
            {
                for (var i = 0; i < oneHotSize; i++)
                {
                    writer[offset] = (i == value) ? 1.0f : 0.0f;
                    offset++;
                }
            }
        }

        public static void WriteZero(this ObservationWriter writer, int offset, int row, int col, int oneHotSize, bool isVisual)
        {
            if (isVisual)
            {
                for (var i = 0; i < oneHotSize; i++)
                {
                    writer[row, col, i] = 0.0f;
                }
            }
            else
            {
                for (var i = 0; i < oneHotSize; i++)
                {
                    writer[offset] = 0.0f;
                    offset++;
                }
            }
        }
    }
}
