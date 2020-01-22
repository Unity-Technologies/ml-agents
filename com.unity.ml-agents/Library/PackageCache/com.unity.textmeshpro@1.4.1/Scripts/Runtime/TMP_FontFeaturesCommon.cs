using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.TextCore.LowLevel;


namespace TMPro
{
    public enum FontFeatureLookupFlags
    {
        IgnoreLigatures             =   0x004,
        IgnoreSpacingAdjustments    =   0x100,
    }

    /// <summary>
    /// The values used to adjust the position of a glyph or set of glyphs.
    /// </summary>
    [Serializable]
    public struct TMP_GlyphValueRecord
    {
        /// <summary>
        /// The positional adjustment affecting the horizontal bearing X of the glyph.
        /// </summary>
        public float xPlacement { get { return m_XPlacement; } set { m_XPlacement = value; } }

        /// <summary>
        /// The positional adjustment affecting the horizontal bearing Y of the glyph.
        /// </summary>
        public float yPlacement { get { return m_YPlacement; } set { m_YPlacement = value; } }

        /// <summary>
        /// The positional adjustment affecting the horizontal advance of the glyph.
        /// </summary>
        public float xAdvance { get { return m_XAdvance; } set { m_XAdvance = value; } }

        /// <summary>
        /// The positional adjustment affecting the vertical advance of the glyph.
        /// </summary>
        public float yAdvance { get { return m_YAdvance; } set { m_YAdvance = value; } }

        // =============================================
        // Private backing fields for public properties.
        // =============================================

        [SerializeField]
        private float m_XPlacement;

        [SerializeField]
        private float m_YPlacement;

        [SerializeField]
        private float m_XAdvance;

        [SerializeField]
        private float m_YAdvance;


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="xPlacement">The positional adjustment affecting the horizontal bearing X of the glyph.</param>
        /// <param name="yPlacement">The positional adjustment affecting the horizontal bearing Y of the glyph.</param>
        /// <param name="xAdvance">The positional adjustment affecting the horizontal advance of the glyph.</param>
        /// <param name="yAdvance">The positional adjustment affecting the vertical advance of the glyph.</param>
        public TMP_GlyphValueRecord(float xPlacement, float yPlacement, float xAdvance, float yAdvance)
        {
            m_XPlacement = xPlacement;
            m_YPlacement = yPlacement;
            m_XAdvance = xAdvance;
            m_YAdvance = yAdvance;
        }

        internal TMP_GlyphValueRecord(GlyphValueRecord_Legacy valueRecord)
        {
            m_XPlacement = valueRecord.xPlacement;
            m_YPlacement = valueRecord.yPlacement;
            m_XAdvance = valueRecord.xAdvance;
            m_YAdvance = valueRecord.yAdvance;
        }

        internal TMP_GlyphValueRecord(GlyphValueRecord valueRecord)
        {
            m_XPlacement = valueRecord.xPlacement;
            m_YPlacement = valueRecord.yPlacement;
            m_XAdvance = valueRecord.xAdvance;
            m_YAdvance = valueRecord.yAdvance;
        }

        public static TMP_GlyphValueRecord operator +(TMP_GlyphValueRecord a, TMP_GlyphValueRecord b)
        {
            TMP_GlyphValueRecord c;
            c.m_XPlacement = a.xPlacement + b.xPlacement;
            c.m_YPlacement = a.yPlacement + b.yPlacement;
            c.m_XAdvance = a.xAdvance + b.xAdvance;
            c.m_YAdvance = a.yAdvance + b.yAdvance;

            return c;
        }
    }

    /// <summary>
    /// The positional adjustment values of a glyph.
    /// </summary>
    [Serializable]
    public struct TMP_GlyphAdjustmentRecord
    {
        /// <summary>
        /// The index of the glyph in the source font file.
        /// </summary>
        public uint glyphIndex { get { return m_GlyphIndex; } set { m_GlyphIndex = value; } }

        /// <summary>
        /// The GlyphValueRecord contains the positional adjustments of the glyph.
        /// </summary>
        public TMP_GlyphValueRecord glyphValueRecord { get { return m_GlyphValueRecord; } set { m_GlyphValueRecord = value; } }

        // =============================================
        // Private backing fields for public properties.
        // =============================================

        [SerializeField]
        private uint m_GlyphIndex;

        [SerializeField]
        private TMP_GlyphValueRecord m_GlyphValueRecord;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="glyphIndex">The index of the glyph in the source font file.</param>
        /// <param name="glyphValueRecord">The GlyphValueRecord contains the positional adjustments of the glyph.</param>
        public TMP_GlyphAdjustmentRecord(uint glyphIndex, TMP_GlyphValueRecord glyphValueRecord)
        {
            m_GlyphIndex = glyphIndex;
            m_GlyphValueRecord = glyphValueRecord;
        }

        internal TMP_GlyphAdjustmentRecord(GlyphAdjustmentRecord adjustmentRecord)
        {
            m_GlyphIndex = adjustmentRecord.glyphIndex;
            m_GlyphValueRecord = new TMP_GlyphValueRecord(adjustmentRecord.glyphValueRecord);
        }
    }

    /// <summary>
    /// The positional adjustment values for a pair of glyphs.
    /// </summary>
    [Serializable]
    public class TMP_GlyphPairAdjustmentRecord
    {
        /// <summary>
        /// Contains the positional adjustment values for the first glyph.
        /// </summary>
        public TMP_GlyphAdjustmentRecord firstAdjustmentRecord { get { return m_FirstAdjustmentRecord; } set { m_FirstAdjustmentRecord = value; } }

        /// <summary>
        /// Contains the positional adjustment values for the second glyph.
        /// </summary>
        public TMP_GlyphAdjustmentRecord secondAdjustmentRecord { get { return m_SecondAdjustmentRecord; } set { m_SecondAdjustmentRecord = value; } }

        /// <summary>
        /// 
        /// </summary>
        public FontFeatureLookupFlags featureLookupFlags { get { return m_FeatureLookupFlags; } set { m_FeatureLookupFlags = value; } }

        // =============================================
        // Private backing fields for public properties.
        // =============================================

        [SerializeField]
        private TMP_GlyphAdjustmentRecord m_FirstAdjustmentRecord;

        [SerializeField]
        private TMP_GlyphAdjustmentRecord m_SecondAdjustmentRecord;

        [SerializeField]
        private FontFeatureLookupFlags m_FeatureLookupFlags;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="firstAdjustmentRecord">First glyph adjustment record.</param>
        /// <param name="secondAdjustmentRecord">Second glyph adjustment record.</param>
        public TMP_GlyphPairAdjustmentRecord(TMP_GlyphAdjustmentRecord firstAdjustmentRecord, TMP_GlyphAdjustmentRecord secondAdjustmentRecord)
        {
            m_FirstAdjustmentRecord = firstAdjustmentRecord;
            m_SecondAdjustmentRecord = secondAdjustmentRecord;
        }

        /// <summary>
        /// Internal constructor 
        /// </summary>
        /// <param name="firstAdjustmentRecord"></param>
        /// <param name="secondAdjustmentRecord"></param>
        internal TMP_GlyphPairAdjustmentRecord(GlyphPairAdjustmentRecord glyphPairAdjustmentRecord)
        {
            m_FirstAdjustmentRecord = new TMP_GlyphAdjustmentRecord(glyphPairAdjustmentRecord.firstAdjustmentRecord);
            m_SecondAdjustmentRecord = new TMP_GlyphAdjustmentRecord(glyphPairAdjustmentRecord.secondAdjustmentRecord);
        }
    }

    public struct GlyphPairKey
    {
        public uint firstGlyphIndex;
        public uint secondGlyphIndex;
        public long key;

        public GlyphPairKey(uint firstGlyphIndex, uint secondGlyphIndex)
        {
            this.firstGlyphIndex = firstGlyphIndex;
            this.secondGlyphIndex = secondGlyphIndex;
            key = (long)secondGlyphIndex << 32 | firstGlyphIndex;
        }

        internal GlyphPairKey(TMP_GlyphPairAdjustmentRecord record)
        {
            firstGlyphIndex = record.firstAdjustmentRecord.glyphIndex;
            secondGlyphIndex = record.secondAdjustmentRecord.glyphIndex;
            key = (long)secondGlyphIndex << 32 | firstGlyphIndex;
        }
    }
}
