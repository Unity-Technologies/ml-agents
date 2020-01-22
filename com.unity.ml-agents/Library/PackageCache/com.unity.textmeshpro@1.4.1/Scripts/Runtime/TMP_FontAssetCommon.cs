using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;


namespace TMPro
{
    /// <summary>
    /// Class that contains the basic information about the font.
    /// </summary>
    [Serializable]
    public class FaceInfo_Legacy
    {
        public string Name;
        public float PointSize;
        public float Scale;

        public int CharacterCount;

        public float LineHeight;
        public float Baseline;
        public float Ascender;
        public float CapHeight;
        public float Descender;
        public float CenterLine;

        public float SuperscriptOffset;
        public float SubscriptOffset;
        public float SubSize;

        public float Underline;
        public float UnderlineThickness;

        public float strikethrough;
        public float strikethroughThickness;

        public float TabWidth;

        public float Padding;
        public float AtlasWidth;
        public float AtlasHeight;
    }


    // Class which contains the Glyph Info / Character definition for each character contained in the font asset.
    [Serializable]
    public class TMP_Glyph : TMP_TextElement_Legacy
    {
        /// <summary>
        /// Function to create a deep copy of a GlyphInfo.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static TMP_Glyph Clone(TMP_Glyph source)
        {
            TMP_Glyph copy = new TMP_Glyph();

            copy.id = source.id;
            copy.x = source.x;
            copy.y = source.y;
            copy.width = source.width;
            copy.height = source.height;
            copy.xOffset = source.xOffset;
            copy.yOffset = source.yOffset;
            copy.xAdvance = source.xAdvance;
            copy.scale = source.scale;

            return copy;
        }
    }


    // Structure which holds the font creation settings
    [Serializable]
    public struct FontAssetCreationSettings
    {
        public string sourceFontFileName;
        public string sourceFontFileGUID;
        public int pointSizeSamplingMode;
        public int pointSize;
        public int padding;
        public int packingMode;
        public int atlasWidth;
        public int atlasHeight;
        public int characterSetSelectionMode;
        public string characterSequence;
        public string referencedFontAssetGUID;
        public string referencedTextAssetGUID;
        public int fontStyle;
        public float fontStyleModifier;
        public int renderMode;
        public bool includeFontFeatures;

        internal FontAssetCreationSettings(string sourceFontFileGUID, int pointSize, int pointSizeSamplingMode, int padding, int packingMode, int atlasWidth, int atlasHeight, int characterSelectionMode, string characterSet, int renderMode)
        {
            this.sourceFontFileName = string.Empty;
            this.sourceFontFileGUID = sourceFontFileGUID;
            this.pointSize = pointSize;
            this.pointSizeSamplingMode = pointSizeSamplingMode;
            this.padding = padding;
            this.packingMode = packingMode;
            this.atlasWidth = atlasWidth;
            this.atlasHeight = atlasHeight;
            this.characterSequence = characterSet;
            this.characterSetSelectionMode = characterSelectionMode;
            this.renderMode = renderMode;

            this.referencedFontAssetGUID = string.Empty;
            this.referencedTextAssetGUID = string.Empty;
            this.fontStyle = 0;
            this.fontStyleModifier = 0;
            this.includeFontFeatures = false;
        }
    }

    /// <summary>
    /// Contains the font assets for the regular and italic styles associated with a given font weight.
    /// </summary>
    [Serializable]
    public struct TMP_FontWeightPair
    {
        public TMP_FontAsset regularTypeface;
        public TMP_FontAsset italicTypeface;
    }


    public struct KerningPairKey
    {
        public uint ascii_Left;
        public uint ascii_Right;
        public uint key;

        public KerningPairKey(uint ascii_left, uint ascii_right)
        {
            ascii_Left = ascii_left;
            ascii_Right = ascii_right;
            key = (ascii_right << 16) + ascii_left;
        }
    }

    /// <summary>
    /// Positional adjustments of a glyph
    /// </summary>
    [Serializable]
    public struct GlyphValueRecord_Legacy
    {
        public float xPlacement;
        public float yPlacement;
        public float xAdvance;
        public float yAdvance;

        internal GlyphValueRecord_Legacy(UnityEngine.TextCore.LowLevel.GlyphValueRecord valueRecord)
        {
            this.xPlacement = valueRecord.xPlacement;
            this.yPlacement = valueRecord.yPlacement;
            this.xAdvance = valueRecord.xAdvance;
            this.yAdvance = valueRecord.yAdvance;
        }

        public static GlyphValueRecord_Legacy operator +(GlyphValueRecord_Legacy a, GlyphValueRecord_Legacy b)
        {
            GlyphValueRecord_Legacy c;
            c.xPlacement = a.xPlacement + b.xPlacement;
            c.yPlacement = a.yPlacement + b.yPlacement;
            c.xAdvance = a.xAdvance + b.xAdvance;
            c.yAdvance = a.yAdvance + b.yAdvance;

            return c;
        }
    }

    [Serializable]
    public class KerningPair
    {
        /// <summary>
        /// The first glyph part of a kerning pair.
        /// </summary>
        public uint firstGlyph
        {
            get { return m_FirstGlyph; }
            set { m_FirstGlyph = value; }
        }
        [FormerlySerializedAs("AscII_Left")]
        [SerializeField]
        private uint m_FirstGlyph;

        /// <summary>
        /// The positional adjustment of the first glyph.
        /// </summary>
        public GlyphValueRecord_Legacy firstGlyphAdjustments
        {
            get { return m_FirstGlyphAdjustments; }
        }
        [SerializeField]
        private GlyphValueRecord_Legacy m_FirstGlyphAdjustments;

        /// <summary>
        /// The second glyph part of a kerning pair.
        /// </summary>
        public uint secondGlyph
        {
            get { return m_SecondGlyph; }
            set { m_SecondGlyph = value; }
        }
        [FormerlySerializedAs("AscII_Right")]
        [SerializeField]
        private uint m_SecondGlyph;

        /// <summary>
        /// The positional adjustment of the second glyph.
        /// </summary>
        public GlyphValueRecord_Legacy secondGlyphAdjustments
        {
            get { return m_SecondGlyphAdjustments; }
        }
        [SerializeField]
        private GlyphValueRecord_Legacy m_SecondGlyphAdjustments;

        [FormerlySerializedAs("XadvanceOffset")]
        public float xOffset;

        internal static KerningPair empty = new KerningPair(0, new GlyphValueRecord_Legacy(), 0, new GlyphValueRecord_Legacy());

        /// <summary>
        /// Determines if the Character Spacing property of the text object will affect the kerning pair.
        /// This is mostly relevant when using Diacritical marks to prevent Character Spacing from altering the 
        /// </summary>
        public bool ignoreSpacingAdjustments
        {
            get { return m_IgnoreSpacingAdjustments; }
        }
        [SerializeField]
        private bool m_IgnoreSpacingAdjustments = false;

        public KerningPair()
        {
            m_FirstGlyph = 0;
            m_FirstGlyphAdjustments = new GlyphValueRecord_Legacy();

            m_SecondGlyph = 0;
            m_SecondGlyphAdjustments = new GlyphValueRecord_Legacy();
        }

        public KerningPair(uint left, uint right, float offset)
        {
            firstGlyph = left;
            m_SecondGlyph = right;
            xOffset = offset;
        }

        public KerningPair(uint firstGlyph, GlyphValueRecord_Legacy firstGlyphAdjustments, uint secondGlyph, GlyphValueRecord_Legacy secondGlyphAdjustments)
        {
            m_FirstGlyph = firstGlyph;
            m_FirstGlyphAdjustments = firstGlyphAdjustments;
            m_SecondGlyph = secondGlyph;
            m_SecondGlyphAdjustments = secondGlyphAdjustments;
        }

        internal void ConvertLegacyKerningData()
        {
            m_FirstGlyphAdjustments.xAdvance = xOffset;
            //xOffset = 0;
        }

    }

    [Serializable]
    public class KerningTable
    {
        public List<KerningPair> kerningPairs;

        public KerningTable()
        {
            kerningPairs = new List<KerningPair>();
        }


        public void AddKerningPair()
        {
            if (kerningPairs.Count == 0)
            {
                kerningPairs.Add(new KerningPair(0, 0, 0));
            }
            else
            {
                uint left = kerningPairs.Last().firstGlyph;
                uint right = kerningPairs.Last().secondGlyph;
                float xoffset = kerningPairs.Last().xOffset;

                kerningPairs.Add(new KerningPair(left, right, xoffset));
            }
        }


        /// <summary>
        /// Add Kerning Pair
        /// </summary>
        /// <param name="first">First glyph</param>
        /// <param name="second">Second glyph</param>
        /// <param name="offset">xAdvance value</param>
        /// <returns></returns>
        public int AddKerningPair(uint first, uint second, float offset)
        {
            int index = kerningPairs.FindIndex(item => item.firstGlyph == first && item.secondGlyph == second);

            if (index == -1)
            {
                kerningPairs.Add(new KerningPair(first, second, offset));
                return 0;
            }

            // Return -1 if Kerning Pair already exists.
            return -1;
        }

        /// <summary>
        /// Add Glyph pair adjustment record
        /// </summary>
        /// <param name="firstGlyph">The first glyph</param>
        /// <param name="firstGlyphAdjustments">Adjustment record for the first glyph</param>
        /// <param name="secondGlyph">The second glyph</param>
        /// <param name="secondGlyphAdjustments">Adjustment record for the second glyph</param>
        /// <returns></returns>
        public int AddGlyphPairAdjustmentRecord(uint first, GlyphValueRecord_Legacy firstAdjustments, uint second, GlyphValueRecord_Legacy secondAdjustments)
        {
            int index = kerningPairs.FindIndex(item => item.firstGlyph == first && item.secondGlyph == second);

            if (index == -1)
            {
                kerningPairs.Add(new KerningPair(first, firstAdjustments, second, secondAdjustments));
                return 0;
            }

            // Return -1 if Kerning Pair already exists.
            return -1;
        }

        public void RemoveKerningPair(int left, int right)
        {
            int index = kerningPairs.FindIndex(item => item.firstGlyph == left && item.secondGlyph == right);

            if (index != -1)
                kerningPairs.RemoveAt(index);
        }


        public void RemoveKerningPair(int index)
        {
            kerningPairs.RemoveAt(index);
        }


        public void SortKerningPairs()
        {
            // Sort List of Kerning Info
            if (kerningPairs.Count > 0)
                kerningPairs = kerningPairs.OrderBy(s => s.firstGlyph).ThenBy(s => s.secondGlyph).ToList();
        }
    }


    public static class TMP_FontUtilities
    {
        private static List<int> k_searchedFontAssets;

        /// <summary>
        /// Search through the given font and its fallbacks for the specified character.
        /// </summary>
        /// <param name="font">The font asset to search for the given character.</param>
        /// <param name="unicode">The character to find.</param>
        /// <param name="character">out parameter containing the glyph for the specified character (if found).</param>
        /// <returns></returns>
        public static TMP_FontAsset SearchForCharacter(TMP_FontAsset font, uint unicode, out TMP_Character character)
        {
            if (k_searchedFontAssets == null)
                k_searchedFontAssets = new List<int>();

            k_searchedFontAssets.Clear();

            return SearchForCharacterInternal(font, unicode, out character);
        }


        /// <summary>
        /// Search through the given list of fonts and their possible fallbacks for the specified character.
        /// </summary>
        /// <param name="fonts"></param>
        /// <param name="unicode"></param>
        /// <param name="character"></param>
        /// <returns></returns>
        public static TMP_FontAsset SearchForCharacter(List<TMP_FontAsset> fonts, uint unicode, out TMP_Character character)
        {
            return SearchForCharacterInternal(fonts, unicode, out character);
        }


        private static TMP_FontAsset SearchForCharacterInternal(TMP_FontAsset font, uint unicode, out TMP_Character character)
        {
            character = null;

            if (font == null) return null;

            if (font.characterLookupTable.TryGetValue(unicode, out character))
            {
                return font;
            }
            else if (font.fallbackFontAssetTable != null && font.fallbackFontAssetTable.Count > 0)
            {
                for (int i = 0; i < font.fallbackFontAssetTable.Count && character == null; i++)
                {
                    TMP_FontAsset temp = font.fallbackFontAssetTable[i];
                    if (temp == null) continue;

                    int id = temp.GetInstanceID();

                    // Skip over the fallback font asset in the event it is null or if already searched.
                    if (k_searchedFontAssets.Contains(id)) continue;

                    // Add to list of font assets already searched.
                    k_searchedFontAssets.Add(id);

                    temp = SearchForCharacterInternal(temp, unicode, out character);

                    if (temp != null)
                        return temp;
                }
            }

            return null;
        }


        private static TMP_FontAsset SearchForCharacterInternal(List<TMP_FontAsset> fonts, uint unicode, out TMP_Character character)
        {
            character = null;

            if (fonts != null && fonts.Count > 0)
            {
                for (int i = 0; i < fonts.Count; i++)
                {
                    TMP_FontAsset fontAsset = SearchForCharacterInternal(fonts[i], unicode, out character);

                    if (fontAsset != null)
                        return fontAsset;
                }
            }

            return null;
        }
    }
}