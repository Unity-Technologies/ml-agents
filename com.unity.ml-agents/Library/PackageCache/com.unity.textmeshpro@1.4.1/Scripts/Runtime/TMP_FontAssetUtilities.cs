using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;


namespace TMPro
{
    public class TMP_FontAssetUtilities
    {
        private static readonly TMP_FontAssetUtilities s_Instance = new TMP_FontAssetUtilities();

        /// <summary>
        /// Default constructor
        /// </summary>
        static TMP_FontAssetUtilities() { }


        /// <summary>
        /// Get a singleton instance of the Font Asset Utilities class.
        /// </summary>
        public static TMP_FontAssetUtilities instance
        {
            get { return s_Instance; }
        }


        /// <summary>
        /// List containing instance ID of font assets already searched.
        /// </summary>
        private static List<int> k_SearchedFontAssets;


        /// <summary>
        /// Returns the text element (character) for the given unicode value taking into consideration the requested font style and weight.
        /// Function searches the source font asset, its list of font assets assigned as alternative typefaces and potentially its fallbacks.
        /// The font asset out parameter contains a reference to the font asset containing the character.
        /// The typeface type indicates whether the returned font asset is the source font asset, an alternative typeface or fallback font asset.
        /// </summary>
        /// <param name="unicode">The unicode value of the requested character</param>
        /// <param name="sourceFontAsset">The font asset to be searched</param>
        /// <param name="includeFallbacks">Include the fallback font assets in the search</param>
        /// <param name="fontStyle">The font style</param>
        /// <param name="fontWeight">The font weight</param>
        /// <param name="type">Indicates if the OUT font asset is an alternative typeface or fallback font asset</param>
        /// <param name="fontAsset">The font asset that contains the requested character</param>
        /// <returns></returns>
        public static TMP_Character GetCharacterFromFontAsset(uint unicode, TMP_FontAsset sourceFontAsset, bool includeFallbacks, FontStyles fontStyle, FontWeight fontWeight, out bool isAlternativeTypeface, out TMP_FontAsset fontAsset)
        {
            if (includeFallbacks)
            {
                if (k_SearchedFontAssets == null)
                    k_SearchedFontAssets = new List<int>();
                else
                    k_SearchedFontAssets.Clear();
            }

            return GetCharacterFromFontAsset_Internal(unicode, sourceFontAsset, includeFallbacks, fontStyle, fontWeight, out isAlternativeTypeface, out fontAsset);
        }


        /// <summary>
        /// Internal function returning the text element character for the given unicode value taking into consideration the font style and weight.
        /// Function searches the source font asset, list of font assets assigned as alternative typefaces and list of fallback font assets.
        /// </summary>
        private static TMP_Character GetCharacterFromFontAsset_Internal(uint unicode, TMP_FontAsset sourceFontAsset, bool includeFallbacks, FontStyles fontStyle, FontWeight fontWeight, out bool isAlternativeTypeface, out TMP_FontAsset fontAsset)
        {
            fontAsset = null;
            isAlternativeTypeface = false;
            TMP_Character characterData = null;

            #region FONT WEIGHT AND FONT STYLE HANDLING
            // Determine if a font weight or style is used. If so check if an alternative typeface is assigned for the given weight and / or style.
            bool isItalic = (fontStyle & FontStyles.Italic) == FontStyles.Italic;

            if (isItalic || fontWeight != FontWeight.Regular)
            {
                // Get reference to the font weight pairs of the given font asset.
                TMP_FontWeightPair[] fontWeights = sourceFontAsset.fontWeightTable;

                int fontWeightIndex = 4;
                switch (fontWeight)
                {
                    case FontWeight.Thin:
                        fontWeightIndex = 1;
                        break;
                    case FontWeight.ExtraLight:
                        fontWeightIndex = 2;
                        break;
                    case FontWeight.Light:
                        fontWeightIndex = 3;
                        break;
                    case FontWeight.Regular:
                        fontWeightIndex = 4;
                        break;
                    case FontWeight.Medium:
                        fontWeightIndex = 5;
                        break;
                    case FontWeight.SemiBold:
                        fontWeightIndex = 6;
                        break;
                    case FontWeight.Bold:
                        fontWeightIndex = 7;
                        break;
                    case FontWeight.Heavy:
                        fontWeightIndex = 8;
                        break;
                    case FontWeight.Black:
                        fontWeightIndex = 9;
                        break;
                }

                fontAsset = isItalic ? fontWeights[fontWeightIndex].italicTypeface : fontWeights[fontWeightIndex].regularTypeface;

                if (fontAsset != null)
                {
                    if (fontAsset.characterLookupTable.TryGetValue(unicode, out characterData))
                    {
                        isAlternativeTypeface = true;

                        return characterData;
                    }
                    else if (fontAsset.atlasPopulationMode == AtlasPopulationMode.Dynamic)
                    {
                        if (fontAsset.TryAddCharacterInternal(unicode, out characterData))
                        {
                            isAlternativeTypeface = true;

                            return characterData;
                        }

                        // Check if the source font file contains the requested character.
                        //if (TryGetCharacterFromFontFile(unicode, fontAsset, out characterData))
                        //{
                        //    isAlternativeTypeface = true;

                        //    return characterData;
                        //}

                        // If we find the requested character, we add it to the font asset character table
                        // and return its character data.
                        // We also add this character to the list of characters we will need to add to the font atlas.
                        // We assume the font atlas has room otherwise this font asset should not be marked as dynamic.
                        // Alternatively, we could also add multiple pages of font atlas textures (feature consideration).
                    }

                    // At this point, we were not able to find the requested character in the alternative typeface
                    // so we check the source font asset and its potential fallbacks.
                }

            }
            #endregion

            // Search the source font asset for the requested character.
            if (sourceFontAsset.characterLookupTable.TryGetValue(unicode, out characterData))
            {
                // We were able to locate the requested character in the given font asset.
                fontAsset = sourceFontAsset;

                return characterData;
            }
            else if (sourceFontAsset.atlasPopulationMode == AtlasPopulationMode.Dynamic)
            {
                if (sourceFontAsset.TryAddCharacterInternal(unicode, out characterData))
                {
                    fontAsset = sourceFontAsset;

                    return characterData;
                }

                //// Check if the source font file contains the requested character.
                //if (TryGetCharacterFromFontFile(unicode, sourceFontAsset, out characterData))
                //{
                //    fontAsset = sourceFontAsset;

                //    //fontAsset.AddCharacterToRasterList(unicode);

                //    return characterData;
                //}

                // If we find the requested character, we add it to the font asset character table
                // and return its character data.
                // We also add this character to the list of characters we will need to add to the font atlas.
                // We assume the font atlas has room otherwise this font asset should not be marked as dynamic.
                // Alternatively, we could also add multiple pages of font atlas textures (feature consideration)
            }

            // Search fallback font assets if we still don't have a valid character and include fallback is set to true.
            if (characterData == null && includeFallbacks && sourceFontAsset.fallbackFontAssetTable != null)
            {
                // Get reference to the list of fallback font assets.
                List<TMP_FontAsset> fallbackFontAssets = sourceFontAsset.fallbackFontAssetTable;
                int fallbackCount = fallbackFontAssets.Count;

                if (fallbackFontAssets != null && fallbackCount > 0)
                {
                    for (int i = 0; i < fallbackCount && characterData == null; i++)
                    {
                        TMP_FontAsset temp = fallbackFontAssets[i];

                        if (temp == null) continue;

                        int id = temp.GetInstanceID();

                        // Skip over the fallback font asset in the event it is null or if already searched.
                        if (k_SearchedFontAssets.Contains(id))
                            continue;

                        // Add to list of font assets already searched.
                        k_SearchedFontAssets.Add(id);

                        characterData = GetCharacterFromFontAsset_Internal(unicode, temp, includeFallbacks, fontStyle, fontWeight, out isAlternativeTypeface, out fontAsset);

                        if (characterData != null)
                        {
                            return characterData;
                        }
                    }
                }

            }

            return null;
        }


        /// <summary>
        /// Returns the text element (character) for the given unicode value taking into consideration the requested font style and weight.
        /// Function searches the provided list of font assets, the list of font assets assigned as alternative typefaces to them as well as their fallbacks.
        /// The font asset out parameter contains a reference to the font asset containing the character.
        /// The typeface type indicates whether the returned font asset is the source font asset, an alternative typeface or fallback font asset.
        /// </summary>
        /// <param name="unicode">The unicode value of the requested character</param>
        /// <param name="fontAssets">The list of font assets to search</param>
        /// <param name="includeFallbacks">Determines if the fallback of each font assets on the list will be searched</param>
        /// <param name="fontStyle">The font style</param>
        /// <param name="fontWeight">The font weight</param>
        /// <param name="type">Determines if the OUT font asset is an alternative typeface or fallback font asset</param>
        /// <param name="fontAsset">The font asset that contains the requested character</param>
        /// <returns></returns>
        public static TMP_Character GetCharacterFromFontAssets(uint unicode, List<TMP_FontAsset> fontAssets, bool includeFallbacks, FontStyles fontStyle, FontWeight fontWeight, out bool isAlternativeTypeface, out TMP_FontAsset fontAsset)
        {
            isAlternativeTypeface = false;

            // Make sure font asset list is valid
            if (fontAssets == null || fontAssets.Count == 0)
            {
                fontAsset = null;
                return null;
            }

            if (includeFallbacks)
            {
                if (k_SearchedFontAssets == null)
                    k_SearchedFontAssets = new List<int>();
                else
                    k_SearchedFontAssets.Clear();
            }

            int fontAssetCount = fontAssets.Count;

            for (int i = 0; i < fontAssetCount; i++)
            {
                if (fontAssets[i] == null) continue;

                TMP_Character characterData = GetCharacterFromFontAsset_Internal(unicode, fontAssets[i], includeFallbacks, fontStyle, fontWeight, out isAlternativeTypeface, out fontAsset);

                if (characterData != null)
                    return characterData;
            }

            fontAsset = null;

            return null;
        }


        // =====================================================================
        // FONT ENGINE & FONT FILE MANAGEMENT - Fields, Properties and Functions
        // =====================================================================

        private static bool k_IsFontEngineInitialized;


        private static bool TryGetCharacterFromFontFile(uint unicode, TMP_FontAsset fontAsset, out TMP_Character character)
        {
            character = null;

            // Initialize Font Engine library if not already initialized
            if (k_IsFontEngineInitialized == false)
            {
                FontEngineError error = FontEngine.InitializeFontEngine();

                if (error == 0)
                    k_IsFontEngineInitialized = true;
            }

            // Load the font face for the given font asset.
            // TODO: Add manager to keep track of which font faces are currently loaded.
            FontEngine.LoadFontFace(fontAsset.sourceFontFile, fontAsset.faceInfo.pointSize);

            Glyph glyph = null;
            uint glyphIndex = FontEngine.GetGlyphIndex(unicode);

            // Check if glyph is already contained in the font asset as the same glyph might be referenced by multiple character.
            if (fontAsset.glyphLookupTable.TryGetValue(glyphIndex, out glyph))
            {
                character = fontAsset.AddCharacter_Internal(unicode, glyph);

                return true;
            }

            GlyphLoadFlags glyphLoadFlags = ((GlyphRasterModes)fontAsset.atlasRenderMode & GlyphRasterModes.RASTER_MODE_HINTED) == GlyphRasterModes.RASTER_MODE_HINTED ? GlyphLoadFlags.LOAD_RENDER : GlyphLoadFlags.LOAD_RENDER | GlyphLoadFlags.LOAD_NO_HINTING;

            if (FontEngine.TryGetGlyphWithUnicodeValue(unicode, glyphLoadFlags, out glyph))
            {
                // Add new character to font asset (if needed)
                character = fontAsset.AddCharacter_Internal(unicode, glyph);

                return true;
            }

            return false;
        }


        public static bool TryGetGlyphFromFontFile(uint glyphIndex, TMP_FontAsset fontAsset, out Glyph glyph)
        {
            glyph = null;

            // Initialize Font Engine library if not already initialized
            if (k_IsFontEngineInitialized == false)
            {
                FontEngineError error = FontEngine.InitializeFontEngine();

                if (error == 0)
                    k_IsFontEngineInitialized = true;
            }

            // Load the font face for the given font asset.
            // TODO: Add manager to keep track of which font faces are currently loaded.
            FontEngine.LoadFontFace(fontAsset.sourceFontFile, fontAsset.faceInfo.pointSize);

            GlyphLoadFlags glyphLoadFlags = ((GlyphRasterModes)fontAsset.atlasRenderMode & GlyphRasterModes.RASTER_MODE_HINTED) == GlyphRasterModes.RASTER_MODE_HINTED ? GlyphLoadFlags.LOAD_RENDER : GlyphLoadFlags.LOAD_RENDER | GlyphLoadFlags.LOAD_NO_HINTING;

            if (FontEngine.TryGetGlyphWithIndexValue(glyphIndex, glyphLoadFlags, out glyph))
            {
                // Add new glyph to font asset (if needed)
                //fontAsset.AddGlyph_Internal(glyph);

                return true;
            }

            return false;
        }

    }
}
