using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
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

    public class Match3SensorNew : ISensor, IBuiltInSensor
    {
        private Match3ObservationType m_ObservationType;
        private ObservationSpec m_ObservationSpec;
        private string m_Name;

        private int m_Rows;
        private int m_Columns;
        private GridValueProvider m_GridValues;
        private int m_OneHotSize;

        public Match3SensorNew(AbstractBoard board, GridValueProvider gvp, int oneHotSize, Match3ObservationType obsType, string name)
        {
            m_Name = name;
            m_Rows = board.Rows;
            m_Columns = board.Columns;
            m_GridValues = gvp;
            m_OneHotSize = oneHotSize;

            m_ObservationType = obsType;
            m_ObservationSpec = obsType == Match3ObservationType.Vector
                ? ObservationSpec.Vector(m_Rows * m_Columns * oneHotSize)
                : ObservationSpec.Visual(m_Rows, m_Columns, oneHotSize);
        }

        public static Match3SensorNew CellTypeSensor(AbstractBoard board, Match3ObservationType obsType, string name)
        {
            return new Match3SensorNew(board, board.GetCellType, board.NumCellTypes, obsType, name);
        }

        public static Match3SensorNew SpecialTypeSensor(AbstractBoard board, Match3ObservationType obsType, string name)
        {
            var specialSize = board.NumSpecialTypes == 0 ? 0 : board.NumSpecialTypes + 1;
            return new Match3SensorNew(board, board.GetSpecialType, specialSize, obsType, name);
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            // if (m_Board.Rows != m_Rows || m_Board.Columns != m_Columns || m_Board.NumCellTypes != m_NumCellTypes)
            // {
            //     Debug.LogWarning(
            //         $"Board shape changes since sensor initialization. This may cause unexpected results. " +
            //         $"Old shape: Rows={m_Rows} Columns={m_Columns}, NumCellTypes={m_NumCellTypes} " +
            //         $"Current shape: Rows={m_Board.Rows} Columns={m_Board.Columns}, NumCellTypes={m_Board.NumCellTypes}"
            //     );
            // }

            if (m_ObservationType == Match3ObservationType.Vector)
            {
                int offset = 0;
                for (var r = 0; r < m_Rows; r++)
                {
                    for (var c = 0; c < m_Columns; c++)
                    {
                        var val = m_GridValues(r, c);

                        for (var i = 0; i < m_OneHotSize; i++)
                        {
                            writer[offset] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }
                    }
                }

                return offset;
            }
            else
            {
                // TODO combine loops? Only difference is inner-most statement.
                int offset = 0;
                for (var r = 0; r < m_Rows; r++)
                {
                    for (var c = 0; c < m_Columns; c++)
                    {
                        var val = m_GridValues(r, c);
                        for (var i = 0; i < m_OneHotSize; i++)
                        {
                            writer[r, c, i] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }
                    }
                }

                return offset;
            }
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            var height = m_Rows;
            var width = m_Columns;
            var tempTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            var converter = new OneHotToTextureUtil(height, width);
            var bytesOut = new List<byte>();

            // Encode the cell types and special types as separate batches of PNGs
            // This is potentially wasteful, e.g. if there are 4 cell types and 1 special type, we could
            // fit in in 2 images, but we'll use 3 here (2 PNGs for the 4 cell type channels, and 1 for
            // the special types). Note that we have to also implement the sparse channel mapping.
            // Optimize this it later.
            var numCellImages = (m_OneHotSize + 2) / 3;
            for (var i = 0; i < numCellImages; i++)
            {
                converter.EncodeToTexture(m_GridValues, tempTexture, 3 * i);
                bytesOut.AddRange(tempTexture.EncodeToPNG());
            }

            DestroyTexture(tempTexture);
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

        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                Object.DestroyImmediate(texture);
            }
            else
            {
                Object.Destroy(texture);
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
        int m_Height;
        int m_Width;
        private static Color[] s_OneHotColors = { Color.red, Color.green, Color.blue };

        public OneHotToTextureUtil(int height, int width)
        {
            m_Colors = new Color[height * width];
            m_Height = height;
            m_Width = width;
        }

        public void EncodeToTexture(GridValueProvider gridValueProvider, Texture2D texture, int channelOffset)
        {
            var i = 0;
            // There's an implicit flip converting to PNG from texture, so make sure we
            // counteract that when forming the texture by iterating through h in reverse.
            for (var h = m_Height - 1; h >= 0; h--)
            {
                for (var w = 0; w < m_Width; w++)
                {
                    int oneHotValue = gridValueProvider(h, w);
                    if (oneHotValue < channelOffset || oneHotValue >= channelOffset + 3)
                    {
                        m_Colors[i++] = Color.black;
                    }
                    else
                    {
                        m_Colors[i++] = s_OneHotColors[oneHotValue - channelOffset];
                    }
                }
            }
            texture.SetPixels(m_Colors);
        }
    }
}
