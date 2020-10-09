using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Match3
{
    public enum Match3ObservationType
    {
        Vector,
        UncompressedVisual,
        CompressedVisual
    }

    public class Match3Sensor : ISensor
    {
        private Match3ObservationType m_ObservationType;
        private AbstractBoard m_Board;
        private int[] m_Shape;

        private int m_Rows;
        private int m_Columns;
        private int m_NumCellTypes;
        private int m_NumSpecialTypes;

        private int SpecialTypeSize
        {
            get { return m_NumSpecialTypes == 0 ? 0 : m_NumSpecialTypes + 1; }
        }

        public Match3Sensor(AbstractBoard board, Match3ObservationType obsType)
        {
            m_Board = board;
            m_Rows = board.Rows;
            m_Columns = board.Columns;
            m_NumCellTypes = board.NumCellTypes;
            m_NumSpecialTypes = board.NumSpecialTypes;

            m_ObservationType = obsType;
            m_Shape = obsType == Match3ObservationType.Vector ?
                new[] { m_Rows * m_Columns * (m_NumCellTypes + SpecialTypeSize) } :
                new[] { m_Rows, m_Columns, m_NumCellTypes + SpecialTypeSize };

            // See comment in GetCompressedObservation()
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public int Write(ObservationWriter writer)
        {
            if (m_Board.Rows != m_Rows || m_Board.Columns != m_Columns || m_Board.NumCellTypes != m_NumCellTypes)
            {
                Debug.LogWarning(
                    $"Board shape changes since sensor initialization. This may cause unexpected results. " +
                    $"Old shape: Rows={m_Rows} Columns={m_Columns}, NumCellTypes={m_NumCellTypes} " +
                    $"Current shape: Rows={m_Board.Rows} Columns={m_Board.Columns}, NumCellTypes={m_Board.NumCellTypes}"
                );
            }

            if (m_ObservationType == Match3ObservationType.Vector)
            {
                int offset = 0;
                for (var r = 0; r < m_Rows; r++)
                {
                    for (var c = 0; c < m_Columns; c++)
                    {
                        var val = m_Board.GetCellType(r, c);
                        for (var i = 0; i < m_NumCellTypes; i++)
                        {
                            writer[offset] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }

                        if (m_NumSpecialTypes > 0)
                        {
                            var special = m_Board.GetSpecialType(r, c);
                            for (var i = 0; i < SpecialTypeSize; i++)
                            {
                                writer[offset] = (i == special) ? 1.0f : 0.0f;
                                offset++;
                            }
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
                        var val = m_Board.GetCellType(r, c);
                        for (var i = 0; i < m_NumCellTypes; i++)
                        {
                            writer[r, c, i] = (i == val) ? 1.0f : 0.0f;
                            offset++;
                        }

                        if (m_NumSpecialTypes > 0)
                        {
                            var special = m_Board.GetSpecialType(r, c);
                            for (var i = 0; i < SpecialTypeSize; i++)
                            {
                                writer[offset] = (i == special) ? 1.0f : 0.0f;
                                offset++;
                            }
                        }
                    }
                }

                return offset;
            }
        }

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
            var numCellImages = (m_NumCellTypes + 2) / 3;
            for (var i = 0; i < numCellImages; i++)
            {
                converter.EncodeToTexture(m_Board.GetCellType, tempTexture, 3 * i);
                bytesOut.AddRange(tempTexture.EncodeToPNG());
            }

            var numSpecialImages = (m_NumSpecialTypes + 2) / 3;
            for (var i = 0; i < numSpecialImages; i++)
            {
                converter.EncodeToTexture(m_Board.GetSpecialType, tempTexture, 3 * i);
                bytesOut.AddRange(tempTexture.EncodeToPNG());
            }

            DestroyTexture(tempTexture);
            return bytesOut.ToArray();
        }

        public void Update()
        {
        }

        public void Reset()
        {
        }

        public SensorCompressionType GetCompressionType()
        {
            return m_ObservationType == Match3ObservationType.CompressedVisual ?
                SensorCompressionType.PNG :
                SensorCompressionType.None;
        }

        public string GetName()
        {
            return "Match3 Sensor";
        }

        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                // TODO move to extension methods for UnityEngine.Object?
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
    public class OneHotToTextureUtil
    {
        Color[] m_Colors;
        int m_Height;
        int m_Width;
        private static Color[] s_OneHotColors = { Color.red, Color.green, Color.blue };

        public delegate int GridValueProvider(int x, int y);


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
