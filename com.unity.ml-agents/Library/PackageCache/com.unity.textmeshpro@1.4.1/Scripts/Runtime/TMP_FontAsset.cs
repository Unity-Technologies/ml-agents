using System;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.Profiling;
using System.Collections;
using System.Collections.Generic;
using System.Linq;


namespace TMPro
{
    public enum AtlasPopulationMode
    {
        Static = 0x0,
        Dynamic = 0x1,
    }


    [Serializable]
    public class TMP_FontAsset : TMP_Asset
    {
        /// <summary>
        /// The version of the font asset class.
        /// Version 1.1.0 adds support for the new TextCore.FontEngine and Dynamic SDF system.
        /// </summary>
        public string version
        {
            get { return m_Version; }
            internal set { m_Version = value; }
        }
        [SerializeField]
        private string m_Version;

        /// <summary>
        /// This field is set when the font asset is first created.
        /// </summary>
        [SerializeField]
        internal string m_SourceFontFileGUID;
        
        #if UNITY_EDITOR
        /// <summary>
        /// Persistent reference to the source font file maintained in the editor.
        /// </summary>
        [SerializeField]
        internal Font m_SourceFontFile_EditorRef;
        #endif

        /// <summary>
        /// Source font file when atlas population mode is set to dynamic. Null when the atlas population mode is set to static. 
        /// </summary>
        public Font sourceFontFile
        {
            get { return m_SourceFontFile; }
            internal set { m_SourceFontFile = value; }
        }
        [SerializeField]
        private Font m_SourceFontFile;

        public AtlasPopulationMode atlasPopulationMode
        {
            get { return m_AtlasPopulationMode; }

            set
            {
                m_AtlasPopulationMode = value;

                #if UNITY_EDITOR
                if (m_AtlasPopulationMode == AtlasPopulationMode.Static)
                    m_SourceFontFile = null;
                else if (m_AtlasPopulationMode == AtlasPopulationMode.Dynamic)
                    m_SourceFontFile = m_SourceFontFile_EditorRef;
                #endif
            }
        }
        [SerializeField]
        private AtlasPopulationMode m_AtlasPopulationMode;


        /// <summary>
        /// Information about the font face.
        /// </summary>
        public FaceInfo faceInfo
        {
            get { return m_FaceInfo; }
            internal set { m_FaceInfo = value; }
        }
        [SerializeField]
        private FaceInfo m_FaceInfo;


        /// <summary>
        /// List of glyphs contained in the font asset.
        /// </summary>
        public List<Glyph> glyphTable
        {
            get { return m_GlyphTable; }
            internal set { m_GlyphTable = value; }
        }
        [SerializeField]
        private List<Glyph> m_GlyphTable = new List<Glyph>();

        /// <summary>
        /// Dictionary used to lookup glyphs contained in the font asset by their index.
        /// </summary>
        public Dictionary<uint, Glyph> glyphLookupTable
        {
            get
            {
                if (m_GlyphLookupDictionary == null)
                    ReadFontAssetDefinition();

                return m_GlyphLookupDictionary;
            }
        }
        private Dictionary<uint, Glyph> m_GlyphLookupDictionary;


        /// <summary>
        /// List containing the characters of the given font asset.
        /// </summary>
        public List<TMP_Character> characterTable
        {
            get { return m_CharacterTable; }
            internal set { m_CharacterTable = value; }
        }
        [SerializeField]
        private List<TMP_Character> m_CharacterTable = new List<TMP_Character>();

        /// <summary>
        /// Dictionary used to lookup characters contained in the font asset by their unicode values.
        /// </summary>
        public Dictionary<uint, TMP_Character> characterLookupTable
        {
            get
            {
                if (m_CharacterLookupDictionary == null)
                    ReadFontAssetDefinition();


                return m_CharacterLookupDictionary;
            }
        }
        private Dictionary<uint, TMP_Character> m_CharacterLookupDictionary;


        /// <summary>
        /// The font atlas used by this font asset.
        /// This is always the texture at index [0] of the fontAtlasTextures.
        /// </summary>
        public Texture2D atlasTexture
        {
            get
            {
                if (m_AtlasTexture == null)
                {
                    m_AtlasTexture = atlasTextures[0];
                }

                return m_AtlasTexture;
            }
        }
        private Texture2D m_AtlasTexture;

        /// <summary>
        /// Array of atlas textures that contain the glyphs used by this font asset.
        /// </summary>
        public Texture2D[] atlasTextures
        {
            get
            {
                if (m_AtlasTextures == null)
                {
                    //
                }

                return m_AtlasTextures;
            }

            set
            {
                m_AtlasTextures = value;
            }
        }
        [SerializeField]
        private Texture2D[] m_AtlasTextures;

        /// <summary>
        /// Index of the font atlas texture that still has available space to add new glyphs.
        /// </summary>
        [SerializeField]
        internal int m_AtlasTextureIndex;

        /// <summary>
        /// List of spaces occupied by glyphs in a given texture.
        /// </summary>
        internal List<GlyphRect> usedGlyphRects
        {
            get { return m_UsedGlyphRects; }
            set { m_UsedGlyphRects = value; }
        }
        [SerializeField]
        private List<GlyphRect> m_UsedGlyphRects;

        /// <summary>
        /// List of spaces available in a given texture to add new glyphs.
        /// </summary>
        internal List<GlyphRect> freeGlyphRects
        {
            get { return m_FreeGlyphRects; }
            set { m_FreeGlyphRects = value; }
        }
        [SerializeField]
        private List<GlyphRect> m_FreeGlyphRects;

        /// <summary>
        /// The general information about the font.
        /// This property and FaceInfo_Legacy type are not longer used in version 1.1.0 of the font asset.
        /// </summary>
		[Obsolete("The fontInfo property and underlying type is now obsolete. Please use the faceInfo property and FaceInfo type instead.")]
        public FaceInfo_Legacy fontInfo
        {
            get { return m_fontInfo; }
        }

        [SerializeField]
        private FaceInfo_Legacy m_fontInfo = null;

        /// <summary>
        /// 
        /// </summary>
        [SerializeField]
        public Texture2D atlas; // Should add a property to make this read-only.

        /// <summary>
        /// The width of the atlas texture(s) used by this font asset.
        /// </summary>
        public int atlasWidth
        {
            get { return m_AtlasWidth; }
            internal set { m_AtlasWidth = value; }
        }
        [SerializeField]
        private int m_AtlasWidth;

        /// <summary>
        /// The height of the atlas texture(s) used by this font asset.
        /// </summary>
        public int atlasHeight
        {
            get { return m_AtlasHeight; }
            internal set { m_AtlasHeight = value; }
        }
        [SerializeField]
        private int m_AtlasHeight;

        /// <summary>
        /// The padding used between glyphs contained in the atlas texture(s) used by this font asset.
        /// </summary>
        public int atlasPadding
        {
            get { return m_AtlasPadding; }
            internal set { m_AtlasPadding = value; }
        }
        [SerializeField]
        private int m_AtlasPadding;

        public GlyphRenderMode atlasRenderMode
        {
            get { return m_AtlasRenderMode; }
            internal set { m_AtlasRenderMode = value; }
        }
        [SerializeField]
        private GlyphRenderMode m_AtlasRenderMode;

        // Legacy field that will eventually be removed.
        [SerializeField]
        internal List<TMP_Glyph> m_glyphInfoList;

        [SerializeField]
        [FormerlySerializedAs("m_kerningInfo")]
        internal KerningTable m_KerningTable = new KerningTable();

        /// <summary>
        /// Table containing the various font features of this font asset. 
        /// </summary>
        public TMP_FontFeatureTable fontFeatureTable
        {
            get { return m_FontFeatureTable; }
            internal set { m_FontFeatureTable = value; }
        }
        [SerializeField]
        private TMP_FontFeatureTable m_FontFeatureTable = new TMP_FontFeatureTable();

        // Legacy field that will eventually be removed
        [SerializeField]
        #pragma warning disable 0649
        private List<TMP_FontAsset> fallbackFontAssets;

        /// <summary>
        /// List containing the Fallback font assets for this font.
        /// </summary>
        public List<TMP_FontAsset> fallbackFontAssetTable
        {
            get { return m_FallbackFontAssetTable; }
            set { m_FallbackFontAssetTable = value; }
        }
        [SerializeField]
        public List<TMP_FontAsset> m_FallbackFontAssetTable;

        /// <summary>
        /// The settings used in the Font Asset Creator when this font asset was created or edited.
        /// </summary>
        public FontAssetCreationSettings creationSettings
        {
            get { return m_CreationSettings; }
            set { m_CreationSettings = value; }
        }
        [SerializeField]
        internal FontAssetCreationSettings m_CreationSettings;

        /// <summary>
        /// Array containing font assets to be used as alternative typefaces for the various potential font weights of this font asset.
        /// </summary>
        public TMP_FontWeightPair[] fontWeightTable
        {
            get { return m_FontWeightTable; }
            internal set { m_FontWeightTable = value; }
        }
        [SerializeField]
        private TMP_FontWeightPair[] m_FontWeightTable = new TMP_FontWeightPair[10];

        // FONT WEIGHTS
        /// <summary>
        /// Font weights used by font asset prior to version 1.1.0.
        /// This is legacy and will be removed at some point in the future.
        /// </summary>
        [SerializeField]
        private TMP_FontWeightPair[] fontWeights = null;

        //private int[] m_characterSet; // Array containing all the characters in this font asset.

        /// <summary>
        /// Defines the dilation of the text when using regular style.
        /// </summary>
        public float normalStyle = 0;

        /// <summary>
        /// The spacing between characters when using regular style.
        /// </summary>
        public float normalSpacingOffset = 0;

        /// <summary>
        /// Defines the dilation of the text when using bold style.
        /// </summary>
        public float boldStyle = 0.75f;

        /// <summary>
        /// The spacing between characters when using regular style.
        /// </summary>
        public float boldSpacing = 7f;

        /// <summary>
        /// Defines the slant of the text when using italic style.
        /// </summary>
        public byte italicStyle = 35;

        public byte tabSize = 10;

        private byte m_oldTabSize;
        internal bool m_IsFontAssetLookupTablesDirty = false;

        /// <summary>
        /// Create new instance of a font asset using default settings.
        /// </summary>
        /// <param name="font"></param>
        /// <returns></returns>
        public static TMP_FontAsset CreateFontAsset(Font font)
        {
            return CreateFontAsset(font, 90, 9, GlyphRenderMode.SDFAA, 1024, 1024, AtlasPopulationMode.Dynamic);
        }

        /// <summary>
        /// Create new instance of a font asset.
        /// </summary>
        /// <param name="font">The source font file.</param>
        /// <param name="samplingPointSize">The sampling point size.</param>
        /// <param name="atlasPadding">The padding / spread between individual glyphs in the font asset.</param>
        /// <param name="renderMode"></param>
        /// <param name="atlasWidth">The atlas texture width.</param>
        /// <param name="atlasHeight">The atlas texture height.</param>
        /// <param name="atlasPopulationMode"></param>
        /// <returns></returns>
        public static TMP_FontAsset CreateFontAsset(Font font, int samplingPointSize, int atlasPadding, GlyphRenderMode renderMode, int atlasWidth, int atlasHeight, AtlasPopulationMode atlasPopulationMode = AtlasPopulationMode.Dynamic)
        {
            TMP_FontAsset fontAsset = ScriptableObject.CreateInstance<TMP_FontAsset>();

            fontAsset.m_Version = "1.1.0";

            // Set face information
            FontEngine.InitializeFontEngine();
            FontEngine.LoadFontFace(font, samplingPointSize);

            fontAsset.faceInfo = FontEngine.GetFaceInfo();

            // Set font reference and GUID
            if (atlasPopulationMode == AtlasPopulationMode.Dynamic)
                fontAsset.sourceFontFile = font;

            // Set persistent reference to source font file in the Editor only.
            #if UNITY_EDITOR
            string guid;
            long localID;

            UnityEditor.AssetDatabase.TryGetGUIDAndLocalFileIdentifier(font, out guid, out localID);
            fontAsset.m_SourceFontFileGUID = guid;
            fontAsset.m_SourceFontFile_EditorRef = font;
            #endif

            fontAsset.atlasPopulationMode = atlasPopulationMode;

            fontAsset.atlasWidth = atlasWidth;
            fontAsset.atlasHeight = atlasHeight;
            fontAsset.atlasPadding = atlasPadding;
            fontAsset.atlasRenderMode = renderMode;

            // Initialize array for the font atlas textures.
            fontAsset.atlasTextures = new Texture2D[1];

            // Create and add font atlas texture.
            Texture2D texture = new Texture2D(0, 0, TextureFormat.Alpha8, false);

            //texture.name = assetName + " Atlas";
            fontAsset.atlasTextures[0] = texture;

            // Add free rectangle of the size of the texture.
            int packingModifier;
            if (((GlyphRasterModes)renderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP)
            {
                packingModifier = 0;

                // Optimize by adding static ref to shader.
                Material tmp_material = new Material(ShaderUtilities.ShaderRef_MobileBitmap);

                //tmp_material.name = texture.name + " Material";
                tmp_material.SetTexture(ShaderUtilities.ID_MainTex, texture);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureWidth, atlasWidth);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureHeight, atlasHeight);

                fontAsset.material = tmp_material;
            }
            else
            {
                packingModifier = 1;

                // Optimize by adding static ref to shader.
                Material tmp_material = new Material(ShaderUtilities.ShaderRef_MobileSDF);

                //tmp_material.name = texture.name + " Material";
                tmp_material.SetTexture(ShaderUtilities.ID_MainTex, texture);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureWidth, atlasWidth);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureHeight, atlasHeight);

                tmp_material.SetFloat(ShaderUtilities.ID_GradientScale, atlasPadding + packingModifier);

                tmp_material.SetFloat(ShaderUtilities.ID_WeightNormal, fontAsset.normalStyle);
                tmp_material.SetFloat(ShaderUtilities.ID_WeightBold, fontAsset.boldStyle);

                fontAsset.material = tmp_material;
            }

            fontAsset.freeGlyphRects = new List<GlyphRect>() { new GlyphRect(0, 0, atlasWidth - packingModifier, atlasHeight - packingModifier) };
            fontAsset.usedGlyphRects = new List<GlyphRect>();

            // TODO: Consider adding support for extracting glyph positioning data

            fontAsset.ReadFontAssetDefinition();

            return fontAsset;
        }


        void Awake()
        {
            //Debug.Log("TMP Font Asset [" + this.name + "] with Version #" + m_Version + " has been enabled!");

            // Check version number of font asset to see if it needs to be upgraded.
            if (this.material != null && string.IsNullOrEmpty(m_Version))
                UpgradeFontAsset();
        }


        #if UNITY_EDITOR
        void OnValidate()
        {
            //if (m_oldTabSize != tabSize)
            //{
            //    m_oldTabSize = tabSize;
            //    ReadFontAssetDefinition();
            //}

            // Handle changes to atlas population mode
            //if (m_AtlasPopulationMode == AtlasPopulationMode.Static)
            //    m_SourceFontFile = null;
            //else
            //{
            //    string path = UnityEditor.AssetDatabase.GUIDToAssetPath(m_SourceFontFileGUID);

            //    if (path != string.Empty)
            //        m_SourceFontFile = UnityEditor.AssetDatabase.LoadAssetAtPath<Font>(path);
            //}
        }
        #endif

        /// <summary>
        /// Read the various data tables of the font asset to populate its different dictionaries to allow for faster lookup of related font asset data.
        /// </summary>
        internal void InitializeDictionaryLookupTables()
        {
            // Create new instance of the glyph lookup dictionary or clear the existing one.
            if (m_GlyphLookupDictionary == null)
                m_GlyphLookupDictionary = new Dictionary<uint, Glyph>();
            else
                m_GlyphLookupDictionary.Clear();

            int glyphCount = m_GlyphTable.Count;

            // Initialize glyph index array or clear the existing one.
            if (m_GlyphIndexList == null)
                m_GlyphIndexList = new List<uint>();
            else
                m_GlyphIndexList.Clear();

            // Add the characters contained in the character table into the dictionary for faster lookup.
            for (int i = 0; i < glyphCount; i++)
            {
                Glyph glyph = m_GlyphTable[i];

                uint index = glyph.index;

                // TODO: Not sure it is necessary to check here.
                if (m_GlyphLookupDictionary.ContainsKey(index) == false)
                {
                    m_GlyphLookupDictionary.Add(index, glyph);
                    m_GlyphIndexList.Add(index);
                }
            }

            // Create new instance of the character lookup dictionary or clear the existing one.
            if (m_CharacterLookupDictionary == null)
                m_CharacterLookupDictionary = new Dictionary<uint, TMP_Character>();
            else
                m_CharacterLookupDictionary.Clear();

            // Add the characters contained in the character table into the dictionary for faster lookup.
            for (int i = 0; i < m_CharacterTable.Count; i++)
            {
                TMP_Character character = m_CharacterTable[i];

                uint unicode = character.unicode;
                uint glyphIndex = character.glyphIndex;

                if (m_CharacterLookupDictionary.ContainsKey(unicode) == false)
                    m_CharacterLookupDictionary.Add(unicode, character);

                if (m_GlyphLookupDictionary.ContainsKey(glyphIndex))
                {
                    character.glyph = m_GlyphLookupDictionary[glyphIndex];
                }
            }

            // Upgrade Glyph Adjustment Table to the new Font Feature table and Glyph Pair Adjustment Records
            if (m_KerningTable != null && m_KerningTable.kerningPairs != null && m_KerningTable.kerningPairs.Count > 0)
                UpgradeGlyphAdjustmentTableToFontFeatureTable();

            // Read Font Features which will include kerning data.
            if (m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary == null)
                m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary = new Dictionary<long, TMP_GlyphPairAdjustmentRecord>();
            else
                m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.Clear();

            List<TMP_GlyphPairAdjustmentRecord> glyphPairAdjustmentRecords = m_FontFeatureTable.m_GlyphPairAdjustmentRecords;
            if (glyphPairAdjustmentRecords != null)
            {
                for (int i = 0; i < glyphPairAdjustmentRecords.Count; i++)
                {
                    TMP_GlyphPairAdjustmentRecord record = glyphPairAdjustmentRecords[i];

                    long key = new GlyphPairKey(record).key;

                    m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.Add(key, record);
                }
            }
        }


        /// <summary>
        /// 
        /// </summary>
        public void ReadFontAssetDefinition()
        {
            //Debug.Log("Reading Font Definition for " + this.name + ".");

            // Check version number of font asset to see if it needs to be upgraded.
            if (this.material != null && string.IsNullOrEmpty(m_Version))
                UpgradeFontAsset();

            // Initialize lookup tables for characters and glyphs.
            InitializeDictionaryLookupTables();

            // Add Tab char(9) to Dictionary.
            if (m_CharacterLookupDictionary.ContainsKey(9) == false)
            {
                Glyph glyph = new Glyph(0, new GlyphMetrics(0, 0, 0, 0, m_FaceInfo.tabWidth * tabSize), GlyphRect.zero, 1.0f, 0);
                m_CharacterLookupDictionary.Add(9, new TMP_Character(9, glyph));
            }

            // Add Linefeed LF char(10) and Carriage Return CR char(13)
            if (m_CharacterLookupDictionary.ContainsKey(10) == false)
            {
                Glyph glyph = new Glyph(0, new GlyphMetrics(10, 0, 0, 0, 0), GlyphRect.zero, 1.0f, 0);
                m_CharacterLookupDictionary.Add(10, new TMP_Character(10, glyph));

                if (!m_CharacterLookupDictionary.ContainsKey(13))
                    m_CharacterLookupDictionary.Add(13, new TMP_Character(13, glyph));
            }

            // Add Zero Width Space 8203 (0x200B)
            if (m_CharacterLookupDictionary.ContainsKey(8203) == false)
            {
                Glyph glyph = new Glyph(0, new GlyphMetrics(0, 0, 0, 0, 0), GlyphRect.zero, 1.0f, 0);
                m_CharacterLookupDictionary.Add(8203, new TMP_Character(8203, glyph));
            }

            // Add Zero Width Non-Breaking Space 8288 (0x2060)
            if (m_CharacterLookupDictionary.ContainsKey(8288) == false)
            {
                Glyph glyph = new Glyph(0, new GlyphMetrics(0, 0, 0, 0, 0), GlyphRect.zero, 1.0f, 0);
                m_CharacterLookupDictionary.Add(8288, new TMP_Character(8288, glyph));
            }

            // Add Non-Breaking Hyphen 8209 (0x2011)
            if (m_CharacterLookupDictionary.ContainsKey(8209) == false)
            {
                TMP_Character character;

                if (m_CharacterLookupDictionary.TryGetValue(45, out character))
                    m_CharacterLookupDictionary.Add(8209, new TMP_Character(8209, character.glyph));
            }


            // Set Cap Height
            if (m_FaceInfo.capLine == 0 && m_CharacterLookupDictionary.ContainsKey(72))
            {
                uint glyphIndex = m_CharacterLookupDictionary[72].glyphIndex;
                m_FaceInfo.capLine = m_GlyphLookupDictionary[glyphIndex].metrics.horizontalBearingY;
            }

            // Adjust Font Scale for compatibility reasons
            if (m_FaceInfo.scale == 0)
                m_FaceInfo.scale = 1.0f;

            // Set Strikethrough Offset (if needed)
            if (m_FaceInfo.strikethroughOffset == 0)
                m_FaceInfo.strikethroughOffset = m_FaceInfo.capLine / 2.5f;

            // Set Padding value for legacy font assets.
            if (m_AtlasPadding == 0)
            {
                if (material.HasProperty(ShaderUtilities.ID_GradientScale))
                    m_AtlasPadding = (int)material.GetFloat(ShaderUtilities.ID_GradientScale) - 1;
            }

            // Compute Hashcode for the font asset name
            hashCode = TMP_TextUtilities.GetSimpleHashCode(this.name);

            // Compute Hashcode for the material name
            materialHashCode = TMP_TextUtilities.GetSimpleHashCode(material.name);

            m_IsFontAssetLookupTablesDirty = false;
        }


        /// <summary>
        /// Sort the Character table by Unicode values.
        /// </summary>
        internal void SortCharacterTable()
        {
            if (m_CharacterTable != null && m_CharacterTable.Count > 0)
                m_CharacterTable = m_CharacterTable.OrderBy(c => c.unicode).ToList();
        }

        /// <summary>
        /// Sort the Glyph table by index values.
        /// </summary>
        internal void SortGlyphTable()
        {
            if (m_GlyphTable != null && m_GlyphTable.Count > 0)
                m_GlyphTable = m_GlyphTable.OrderBy(c => c.index).ToList();
        }

        /// <summary>
        /// Sort both glyph and character tables.
        /// </summary>
        internal void SortGlyphAndCharacterTables()
        {
            SortGlyphTable();
            SortCharacterTable();
        }


        /// <summary>
        /// Function to check if a certain character exists in the font asset.
        /// </summary>
        /// <param name="character"></param>
        /// <returns></returns>
        public bool HasCharacter(int character)
        {
            if (m_CharacterLookupDictionary == null)
                return false;

            if (m_CharacterLookupDictionary.ContainsKey((uint)character))
                return true;

            return false;
        }


        /// <summary>
        /// Function to check if a certain character exists in the font asset.
        /// </summary>
        /// <param name="character"></param>
        /// <returns></returns>
        public bool HasCharacter(char character)
        {
            if (m_CharacterLookupDictionary == null)
                return false;

            if (m_CharacterLookupDictionary.ContainsKey(character))
                return true;

            return false;
        }


        /// <summary>
        /// Function to check if a character is contained in the font asset with the option to also check through fallback font assets.
        /// </summary>
        /// <param name="character"></param>
        /// <param name="searchFallbacks"></param>
        /// <returns></returns>
        public bool HasCharacter(char character, bool searchFallbacks)
        {
            // Read font asset definition if it hasn't already been done.
            if (m_CharacterLookupDictionary == null)
            {
                ReadFontAssetDefinition();

                if (m_CharacterLookupDictionary == null)
                    return false;
            }

            // Check font asset
            if (m_CharacterLookupDictionary.ContainsKey(character))
                return true;

            // Check if font asset is dynamic and if so try to add the requested character to it.
            if (m_AtlasPopulationMode == AtlasPopulationMode.Dynamic)
            {
                TMP_Character returnedCharacter;

                if (TryAddCharacterInternal(character, out returnedCharacter))
                    return true;
            }

            if (searchFallbacks)
            {
                // Check font asset fallbacks
                if (fallbackFontAssetTable != null && fallbackFontAssetTable.Count > 0)
                {
                    for (int i = 0; i < fallbackFontAssetTable.Count && fallbackFontAssetTable[i] != null; i++)
                    {
                        if (fallbackFontAssetTable[i].HasCharacter_Internal(character, searchFallbacks))
                            return true;
                    }
                }

                // Check general fallback font assets.
                if (TMP_Settings.fallbackFontAssets != null && TMP_Settings.fallbackFontAssets.Count > 0)
                {
                    for (int i = 0; i < TMP_Settings.fallbackFontAssets.Count && TMP_Settings.fallbackFontAssets[i] != null; i++)
                    {
                        if (TMP_Settings.fallbackFontAssets[i].m_CharacterLookupDictionary == null)
                            TMP_Settings.fallbackFontAssets[i].ReadFontAssetDefinition();

                        if (TMP_Settings.fallbackFontAssets[i].m_CharacterLookupDictionary != null && TMP_Settings.fallbackFontAssets[i].HasCharacter_Internal(character, searchFallbacks))
                            return true;
                    }
                }

                // Check TMP Settings Default Font Asset
                if (TMP_Settings.defaultFontAsset != null)
                {
                    if (TMP_Settings.defaultFontAsset.m_CharacterLookupDictionary == null)
                        TMP_Settings.defaultFontAsset.ReadFontAssetDefinition();

                    if (TMP_Settings.defaultFontAsset.m_CharacterLookupDictionary != null && TMP_Settings.defaultFontAsset.HasCharacter_Internal(character, searchFallbacks))
                        return true;
                }
            }

            return false;
        }


        /// <summary>
        /// Function to check if a character is contained in a font asset with the option to also check through fallback font assets.
        /// This private implementation does not search the fallback font asset in the TMP Settings file.
        /// </summary>
        /// <param name="character"></param>
        /// <param name="searchFallbacks"></param>
        /// <returns></returns>
        bool HasCharacter_Internal(char character, bool searchFallbacks)
        {
            // Read font asset definition if it hasn't already been done.
            if (m_CharacterLookupDictionary == null)
            {
                ReadFontAssetDefinition();

                if (m_CharacterLookupDictionary == null)
                    return false;
            }

            // Check font asset
            if (m_CharacterLookupDictionary.ContainsKey(character))
                return true;

            if (searchFallbacks)
            {
                // Check Font Asset Fallback fonts.
                if (fallbackFontAssetTable != null && fallbackFontAssetTable.Count > 0)
                {
                    for (int i = 0; i < fallbackFontAssetTable.Count && fallbackFontAssetTable[i] != null; i++)
                    {
                        if (fallbackFontAssetTable[i].HasCharacter_Internal(character, searchFallbacks))
                            return true;
                    }
                }
            }

            return false;
        }


        /// <summary>
        /// Function to check if certain characters exists in the font asset. Function returns a list of missing characters.
        /// </summary>
        /// <param name="character"></param>
        /// <returns></returns>
        public bool HasCharacters(string text, out List<char> missingCharacters)
        {
            if (m_CharacterLookupDictionary == null)
            {
                missingCharacters = null;
                return false;
            }

            missingCharacters = new List<char>();

            for (int i = 0; i < text.Length; i++)
            {
                if (!m_CharacterLookupDictionary.ContainsKey(text[i]))
                    missingCharacters.Add(text[i]);
            }

            if (missingCharacters.Count == 0)
                return true;

            return false;
        }


        /// <summary>
        /// Function to check if certain characters exists in the font asset. Function returns false if any characters are missing.
        /// </summary>
        /// <param name="text">String containing the characters to check</param>
        /// <returns></returns>
        public bool HasCharacters(string text)
        {
            if (m_CharacterLookupDictionary == null)
                return false;

            for (int i = 0; i < text.Length; i++)
            {
                if (!m_CharacterLookupDictionary.ContainsKey(text[i]))
                    return false;
            }

            return true;
        }


        /// <summary>
        /// Function to extract all the characters from a font asset.
        /// </summary>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        public static string GetCharacters(TMP_FontAsset fontAsset)
        {
            string characters = string.Empty;

            for (int i = 0; i < fontAsset.characterTable.Count; i++)
            {
                characters += (char)fontAsset.characterTable[i].unicode;
            }

            return characters;
        }


        /// <summary>
        /// Function which returns an array that contains all the characters from a font asset.
        /// </summary>
        /// <param name="fontAsset"></param>
        /// <returns></returns>
        public static int[] GetCharactersArray(TMP_FontAsset fontAsset)
        {
            int[] characters = new int[fontAsset.characterTable.Count];

            for (int i = 0; i < fontAsset.characterTable.Count; i++)
            {
                characters[i] = (int)fontAsset.characterTable[i].unicode;
            }

            return characters;
        }


        // ================================================================================
        // Properties and functions related to character and glyph additions as well as
        // tacking glyphs that need to be added to various font asset atlas textures.
        // ================================================================================

        /// <summary>
        /// Determines if the font asset is already registered to be updated.
        /// </summary>
        //private bool m_IsAlreadyRegisteredForUpdate;

        /// <summary>
        /// List of glyphs that need to be added / packed in atlas texture.
        /// </summary>
        private List<Glyph> m_GlyphsToPack = new List<Glyph>();

        /// <summary>
        /// List of glyphs that have been packed in the atlas texture and ready to be rendered.
        /// </summary>
        private List<Glyph> m_GlyphsPacked = new List<Glyph>();

        /// <summary>
        /// 
        /// </summary>
        private List<Glyph> m_GlyphsToRender = new List<Glyph>();

        /// <summary>
        /// List used in the process of adding new glyphs to the atlas texture.
        /// </summary>
        private List<uint> m_GlyphIndexList = new List<uint>();
        private List<TMP_Character> m_CharactersToAdd = new List<TMP_Character>();

        /// <summary>
        /// Internal static array used to avoid allocations when using the GetGlyphPairAdjustmentTable().
        /// </summary>
        internal static uint[] s_GlyphIndexArray = new uint[16];

        /// <summary>
        /// Internal static list used to track characters that could not be added to the font asset.
        /// </summary>
        internal static List<uint> s_MissingCharacterList = new List<uint>(16);

        /// <summary>
        /// Try adding the characters from the provided string to the font asset.
        /// </summary>
        /// <param name="unicodes">Array that contains the characters to add to the font asset.</param>
        /// <returns>Returns true if all the characters were successfully added to the font asset. Return false otherwise.</returns>
        public bool TryAddCharacters(uint[] unicodes)
        {
            uint[] missingUnicodes;

            return TryAddCharacters(unicodes, out missingUnicodes);
        }

        /// <summary>
        /// Try adding the characters from the provided string to the font asset.
        /// </summary>
        /// <param name="unicodes">Array that contains the characters to add to the font asset.</param>
        /// <param name="missingUnicodes">Array containing the characters that could not be added to the font asset.</param>
        /// <returns>Returns true if all the characters were successfully added to the font asset. Return false otherwise.</returns>
        public bool TryAddCharacters(uint[] unicodes, out uint[] missingUnicodes)
        {
            s_MissingCharacterList.Clear();

            // Make sure font asset is set to dynamic and that we have a valid list of characters.
            if (unicodes == null || unicodes.Length == 0 || m_AtlasPopulationMode == AtlasPopulationMode.Static)
            {
                if (m_AtlasPopulationMode == AtlasPopulationMode.Static)
                    Debug.LogWarning("Unable to add characters to font asset [" + this.name + "] because its AtlasPopulationMode is set to Static.", this);
                else
                {
                    Debug.LogWarning("Unable to add characters to font asset [" + this.name + "] because the provided Unicode list is Null or Empty.", this);
                }

                missingUnicodes = unicodes.ToArray();
                return false;
            }

            Profiler.BeginSample("TMP.TryAddCharacter");

            // Load font face.
            if (FontEngine.LoadFontFace(m_SourceFontFile, m_FaceInfo.pointSize) != FontEngineError.Success)
            {
                Profiler.EndSample();

                missingUnicodes = unicodes.ToArray();
                return false;
            }

            // Clear data structures used to track which glyph needs to be added to atlas texture.
            m_GlyphIndexList.Clear();
            m_CharactersToAdd.Clear();

            bool isMissingCharacters = false;
            int unicodeCount = unicodes.Length;

            for (int i = 0; i < unicodeCount; i++)
            {
                uint unicode = unicodes[i];

                // Check if character is already contained in the character table.
                if (m_CharacterLookupDictionary.ContainsKey(unicode))
                    continue;

                // Get the index of the glyph for this unicode value.
                uint glyphIndex = FontEngine.GetGlyphIndex(unicode);

                if (glyphIndex == 0)
                {
                    isMissingCharacters = true;
                    continue;
                }

                TMP_Character character = new TMP_Character(unicode, glyphIndex);

                // Check if glyph is already contained in the font asset as the same glyph might be referenced by multiple characters.
                if (m_GlyphLookupDictionary.ContainsKey(glyphIndex))
                {
                    character.glyph = m_GlyphLookupDictionary[glyphIndex];
                    m_CharacterTable.Add(character);
                    m_CharacterLookupDictionary.Add(unicode, character);

                    continue;
                }

                m_GlyphIndexList.Add(glyphIndex);
                m_CharactersToAdd.Add(character);
            }

            if (m_GlyphIndexList.Count == 0)
            {
                //Debug.LogWarning("No characters will be added to font asset [" + this.name + "] either because they are already present in the font asset or missing from the font file.");
                Profiler.EndSample();

                missingUnicodes = unicodes.ToArray();
                return false;
            }

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                //Debug.Log("Setting initial size of atlas texture used by font asset [" + this.name + "].");
                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            Glyph[] glyphs;

            bool allCharactersAdded = FontEngine.TryAddGlyphsToTexture(m_GlyphIndexList, m_AtlasPadding, GlyphPackingMode.BestShortSideFit, m_FreeGlyphRects, m_UsedGlyphRects, m_AtlasRenderMode, m_AtlasTextures[m_AtlasTextureIndex], out glyphs);

            // Add new glyphs to relevant data structures.
            for (int i = 0; i < glyphs.Length; i++)
            {
                Glyph glyph = glyphs[i];
                uint glyphIndex = glyph.index;
                
                // Add new glyph to glyph table.
                m_GlyphTable.Add(glyph);
                m_GlyphLookupDictionary.Add(glyphIndex, glyph);
            }

            // Add new characters to relevant data structures.
            for (int i = 0; i < m_CharactersToAdd.Count; i++)
            {
                TMP_Character character = m_CharactersToAdd[i];
                Glyph glyph;

                if (m_GlyphLookupDictionary.TryGetValue(character.glyphIndex, out glyph) == false)
                {
                    s_MissingCharacterList.Add(character.unicode);
                    continue;
                }

                character.glyph = glyph;
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(character.unicode, character);
            }

            #if UNITY_EDITOR
            // Makes the changes to the font asset persistent.
            if (UnityEditor.EditorUtility.IsPersistent(this))
            {
                TMP_EditorResourceManager.RegisterResourceForUpdate(this);
            }
            #endif

            Profiler.EndSample();

            missingUnicodes = null;

            if (s_MissingCharacterList.Count > 0)
                missingUnicodes = s_MissingCharacterList.ToArray();

            return allCharactersAdded && !isMissingCharacters;
        }

        /// <summary>
        /// Try adding the characters from the provided string to the font asset.
        /// </summary>
        /// <param name="characters">String containing the characters to add to the font asset.</param>
        /// <returns>Returns true if all the characters were successfully added to the font asset. Return false otherwise.</returns>
        public bool TryAddCharacters(string characters)
        {
            string missingCharacters;

            return TryAddCharacters(characters, out missingCharacters);
        }


        /// <summary>
        /// Try adding the characters from the provided string to the font asset.
        /// </summary>
        /// <param name="characters">String containing the characters to add to the font asset.</param>
        /// <param name="missingCharacters">String containing the characters that could not be added to the font asset.</param>
        /// <returns>Returns true if all the characters were successfully added to the font asset. Return false otherwise.</returns>
        public bool TryAddCharacters(string characters, out string missingCharacters)
        {
            // Make sure font asset is set to dynamic and that we have a valid list of characters.
            if (string.IsNullOrEmpty(characters) || m_AtlasPopulationMode == AtlasPopulationMode.Static)
            {
                if (m_AtlasPopulationMode == AtlasPopulationMode.Static)
                    Debug.LogWarning("Unable to add characters to font asset [" + this.name + "] because its AtlasPopulationMode is set to Static.", this);
                else
                {
                    Debug.LogWarning("Unable to add characters to font asset [" + this.name + "] because the provided character list is Null or Empty.", this);
                }

                missingCharacters = characters;
                return false;
            }

            // Load font face.
            if (FontEngine.LoadFontFace(m_SourceFontFile, m_FaceInfo.pointSize) != FontEngineError.Success)
            {
                missingCharacters = characters;
                return false;
            }

            // Clear data structures used to track which glyph needs to be added to atlas texture.
            m_GlyphIndexList.Clear();
            m_CharactersToAdd.Clear();

            bool isMissingCharacters = false;
            int characterCount = characters.Length;

            // Iterate over each of the requested characters.
            for (int i = 0; i < characterCount; i++)
            {
                uint unicode = characters[i];

                // Check if character is already contained in the character table.
                if (m_CharacterLookupDictionary.ContainsKey(unicode))
                    continue;

                // Get the index of the glyph for this unicode value.
                uint glyphIndex = FontEngine.GetGlyphIndex(unicode);

                // Skip missing glyphs
                if (glyphIndex == 0)
                {
                    // Might want to keep track and report the missing characters.
                    isMissingCharacters = true;
                    continue;
                }

                TMP_Character character = new TMP_Character(unicode, glyphIndex);

                // Check if glyph is already contained in the font asset as the same glyph might be referenced by multiple characters.
                if (m_GlyphLookupDictionary.ContainsKey(glyphIndex))
                {
                    character.glyph = m_GlyphLookupDictionary[glyphIndex];
                    m_CharacterTable.Add(character);
                    m_CharacterLookupDictionary.Add(unicode, character);

                    continue;
                }

                // Add glyph to list of glyphs to add and glyph lookup map.
                m_GlyphIndexList.Add(glyphIndex);
                m_CharactersToAdd.Add(character);
            }

            if (m_GlyphIndexList.Count == 0)
            {
                missingCharacters = characters;
                return false;
            }

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                //Debug.Log("Setting initial size of atlas texture used by font asset [" + this.name + "].");
                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            Glyph[] glyphs;

            bool allCharactersAdded = FontEngine.TryAddGlyphsToTexture(m_GlyphIndexList, m_AtlasPadding, GlyphPackingMode.BestShortSideFit, m_FreeGlyphRects, m_UsedGlyphRects, m_AtlasRenderMode, m_AtlasTextures[m_AtlasTextureIndex], out glyphs);

            for (int i = 0; i < glyphs.Length; i++)
            {
                Glyph glyph = glyphs[i];
                uint glyphIndex = glyph.index;

                // Add new glyph to glyph table.
                m_GlyphTable.Add(glyph);
                m_GlyphLookupDictionary.Add(glyphIndex, glyph);
            }

            missingCharacters = string.Empty;

            // Add new characters to relevant data structures.
            for (int i = 0; i < m_CharactersToAdd.Count; i++)
            {
                TMP_Character character = m_CharactersToAdd[i];
                Glyph glyph;

                if (m_GlyphLookupDictionary.TryGetValue(character.glyphIndex, out glyph) == false)
                {
                    // TODO: Revise to avoid string concatenation.
                    missingCharacters += (char)character.unicode;
                    continue;
                }

                character.glyph = glyph;
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(character.unicode, character);
            }

            #if UNITY_EDITOR
            // Makes the changes to the font asset persistent.
            if (UnityEditor.EditorUtility.IsPersistent(this))
            {
                TMP_EditorResourceManager.RegisterResourceForUpdate(this);
            }
            #endif

            return allCharactersAdded && !isMissingCharacters;
        }


        /// <summary>
        /// NOT USED CURRENTLY - Try adding character using Unicode value to font asset.
        /// </summary>
        /// <param name="unicode">The Unicode value of the character.</param>
        /// <param name="character">The character data if successfully added to the font asset. Null otherwise.</param>
        /// <returns>Returns true if the character has been added. False otherwise.</returns>
        internal bool TryAddCharacter_Internal(uint unicode)
        {
            TMP_Character character = null;
            
            // Check if character is already contained in the character table.
            if (m_CharacterLookupDictionary.ContainsKey(unicode))
                return true;

            uint glyphIndex = FontEngine.GetGlyphIndex(unicode);
            if (glyphIndex == 0)
                return false;

            // Check if glyph is already contained in the font asset as the same glyph might be referenced by multiple characters.
            if (m_GlyphLookupDictionary.ContainsKey(glyphIndex))
            {
                character = new TMP_Character(unicode, m_GlyphLookupDictionary[glyphIndex]);
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(unicode, character);

                //#if UNITY_EDITOR
                // Makes the changes to the font asset persistent.
                // OPTIMIZATION: This could be handled when exiting Play mode if we added any new characters to the asset.
                // Could also add some update registry to handle this.
                //SortGlyphTable();
                //UnityEditor.EditorUtility.SetDirty(this);
                //#endif

                return true;
            }

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                //Debug.Log("Setting initial size of atlas texture used by font asset [" + this.name + "].");
                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            Glyph glyph;

            if (FontEngine.TryAddGlyphToTexture(glyphIndex, m_AtlasPadding, GlyphPackingMode.BestShortSideFit, m_FreeGlyphRects, m_UsedGlyphRects, m_AtlasRenderMode, m_AtlasTextures[m_AtlasTextureIndex], out glyph))
            {
                // Add new glyph to glyph table.
                m_GlyphTable.Add(glyph);
                m_GlyphLookupDictionary.Add(glyphIndex, glyph);

                // Add new character
                character = new TMP_Character(unicode, glyph);
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(unicode, character);

                //#if UNITY_EDITOR
                // Makes the changes to the font asset persistent.
                // OPTIMIZATION: This could be handled when exiting Play mode if we added any new characters to the asset.
                // Could also add some update registry to handle this.
                //SortGlyphTable();
                //UnityEditor.EditorUtility.SetDirty(this);
                //#endif

                return true;
            }

            return false;
        }


        /// <summary>
        /// To be removed.
        /// </summary>
        /// <param name="unicode"></param>
        /// <param name="glyph"></param>
        internal TMP_Character AddCharacter_Internal(uint unicode, Glyph glyph)
        {
            // Check if character is already contained in the character table.
            if (m_CharacterLookupDictionary.ContainsKey(unicode))
                return m_CharacterLookupDictionary[unicode];

            uint glyphIndex = glyph.index;

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                //Debug.Log("Setting initial size of atlas texture used by font asset [" + this.name + "].");
                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            // Check if glyph is already contained in the glyph table.
            if (m_GlyphLookupDictionary.ContainsKey(glyphIndex) == false)
            {
                if (glyph.glyphRect.width == 0 || glyph.glyphRect.width == 0)
                {
                    // Glyphs with zero width and / or height can be automatically added to font asset.
                    m_GlyphTable.Add(glyph);
                }
                else
                {
                    // Try packing new glyph 
                    if (FontEngine.TryPackGlyphInAtlas(glyph, m_AtlasPadding, GlyphPackingMode.ContactPointRule, m_AtlasRenderMode, m_AtlasWidth, m_AtlasHeight, m_FreeGlyphRects, m_UsedGlyphRects) == false)
                    {
                        // TODO: Add handling to create new atlas texture to fit glyph.

                        return null;
                    }

                    m_GlyphsToRender.Add(glyph);
                }
            }

            // Add character to font asset.
            TMP_Character character = new TMP_Character(unicode, glyph);
            m_CharacterTable.Add(character);
            m_CharacterLookupDictionary.Add(unicode, character);

            //Debug.Log("Adding character [" + (char)unicode + "] with Unicode (" + unicode + ") to [" + this.name + "] font asset.");

            // Schedule glyph to be added to the font atlas texture
            //TM_FontAssetUpdateManager.RegisterFontAssetForUpdate(this);
            UpdateAtlasTexture(); // Temporary until callback system is revised.

            //#if UNITY_EDITOR
            // Makes the changes to the font asset persistent.
            // OPTIMIZATION: This could be handled when exiting Play mode if we added any new characters to the asset.
            // Could also add some update registry to handle this.
            //SortGlyphTable();
            //UnityEditor.EditorUtility.SetDirty(this);
            //#endif

            return character;
        }


        /// <summary>
        /// Try adding character using Unicode value to font asset.
        /// Function assumes internal user has already checked to make sure the character is not already contained in the font asset.
        /// </summary>
        /// <param name="unicode">The Unicode value of the character.</param>
        /// <param name="character">The character data if successfully added to the font asset. Null otherwise.</param>
        /// <returns>Returns true if the character has been added. False otherwise.</returns>
        internal bool TryAddCharacterInternal(uint unicode, out TMP_Character character)
        {
            character = null;

            // Load font face.
            if (FontEngine.LoadFontFace(sourceFontFile, m_FaceInfo.pointSize) != FontEngineError.Success)
                return false;

            uint glyphIndex = FontEngine.GetGlyphIndex(unicode);
            if (glyphIndex == 0)
                return false;

            Profiler.BeginSample("TMP.TryAddCharacter");

            // Check if glyph is already contained in the font asset as the same glyph might be referenced by multiple characters.
            if (m_GlyphLookupDictionary.ContainsKey(glyphIndex))
            {
                character = new TMP_Character(unicode, m_GlyphLookupDictionary[glyphIndex]);
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(unicode, character);

                if (TMP_Settings.getFontFeaturesAtRuntime)
                    UpdateGlyphAdjustmentRecords(unicode, glyphIndex);

                #if UNITY_EDITOR
                // Makes the changes to the font asset persistent.
                // OPTIMIZATION: This could be handled when exiting Play mode if we added any new characters to the asset.
                // Could also add some update registry to handle this.
                //SortGlyphTable();
                if (UnityEditor.EditorUtility.IsPersistent(this))
                {
                    TMP_EditorResourceManager.RegisterResourceForUpdate(this);
                }
                #endif

                Profiler.EndSample();

                return true;
            }

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                // TODO: Need texture to be readable.
                if (m_AtlasTextures[m_AtlasTextureIndex].isReadable == false)
                {
                    Debug.LogWarning("Unable to add the requested character to font asset [" + this.name + "]'s atlas texture. Please make the texture [" + m_AtlasTextures[m_AtlasTextureIndex].name + "] readable.", m_AtlasTextures[m_AtlasTextureIndex]);

                    Profiler.EndSample();
                    return false;
                }

                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            Glyph glyph;

            if (FontEngine.TryAddGlyphToTexture(glyphIndex, m_AtlasPadding, GlyphPackingMode.BestShortSideFit, m_FreeGlyphRects, m_UsedGlyphRects, m_AtlasRenderMode, m_AtlasTextures[m_AtlasTextureIndex], out glyph))
            {
                // Add new glyph to glyph table.
                m_GlyphTable.Add(glyph);
                m_GlyphLookupDictionary.Add(glyphIndex, glyph);

                // Add new character
                character = new TMP_Character(unicode, glyph);
                m_CharacterTable.Add(character);
                m_CharacterLookupDictionary.Add(unicode, character);

                m_GlyphIndexList.Add(glyphIndex);

                if (TMP_Settings.getFontFeaturesAtRuntime)
                    UpdateGlyphAdjustmentRecords(unicode, glyphIndex);

                #if UNITY_EDITOR
                // Makes the changes to the font asset persistent.
                // OPTIMIZATION: This could be handled when exiting Play mode if we added any new characters to the asset.
                // Could also add some update registry to handle this.
                //SortGlyphTable();
                if (UnityEditor.EditorUtility.IsPersistent(this))
                {
                    TMP_EditorResourceManager.RegisterResourceForUpdate(this);
                }
                #endif

                Profiler.EndSample();

                return true;
            }

            Profiler.EndSample();

            return false;
        }


        /// <summary>
        /// Internal function used to get the glyph index for the given unicode.
        /// </summary>
        /// <param name="unicode"></param>
        /// <returns></returns>
        internal uint GetGlyphIndex(uint unicode)
        {
            // Load font face.
            if (FontEngine.LoadFontFace(sourceFontFile, m_FaceInfo.pointSize) != FontEngineError.Success)
                return 0;

            return FontEngine.GetGlyphIndex(unicode);
        }


        internal void UpdateAtlasTexture()
        {
            // Return if we don't have any glyphs to add to atlas texture.
            // This is possible if UpdateAtlasTexture() was called manually.
            //if (m_GlyphsToPack.Count == 0)
            //    return;

            if (m_GlyphsToRender.Count == 0)
                return;

            //Debug.Log("Updating [" + this.name + "]'s atlas texture.");

            // Pack glyphs in the given atlas texture size. 
            // TODO: Packing and glyph render modes should be defined in the font asset.
            //FontEngine.PackGlyphsInAtlas(m_GlyphsToPack, m_GlyphsPacked, m_AtlasPadding, GlyphPackingMode.ContactPointRule, GlyphRenderMode.SDFAA, m_AtlasWidth, m_AtlasHeight, m_FreeGlyphRects, m_UsedGlyphRects);
            //FontEngine.RenderGlyphsToTexture(m_GlyphsPacked, m_AtlasPadding, GlyphRenderMode.SDFAA, m_AtlasTextures[m_AtlasTextureIndex]);

            // Resize the Atlas Texture to the appropriate size
            if (m_AtlasTextures[m_AtlasTextureIndex].width == 0 || m_AtlasTextures[m_AtlasTextureIndex].height == 0)
            {
                //Debug.Log("Setting initial size of atlas texture used by font asset [" + this.name + "].");
                m_AtlasTextures[m_AtlasTextureIndex].Resize(m_AtlasWidth, m_AtlasHeight);
                FontEngine.ResetAtlasTexture(m_AtlasTextures[m_AtlasTextureIndex]);
            }

            FontEngine.RenderGlyphsToTexture(m_GlyphsToRender, m_AtlasPadding, m_AtlasRenderMode, m_AtlasTextures[m_AtlasTextureIndex]);

            // Apply changes to atlas texture
            m_AtlasTextures[m_AtlasTextureIndex].Apply(false, false);

            // Add glyphs that were successfully packed to the glyph table.
            for (int i = 0; i < m_GlyphsToRender.Count /* m_GlyphsPacked.Count */; i++)
            {
                Glyph glyph = m_GlyphsToRender[i]; // m_GlyphsPacked[i];

                // Update atlas texture index
                glyph.atlasIndex = m_AtlasTextureIndex;

                m_GlyphTable.Add(glyph);
                m_GlyphLookupDictionary.Add(glyph.index, glyph);
            }

            // Clear list of glyphs
            m_GlyphsPacked.Clear();
            m_GlyphsToRender.Clear();

            // Add any remaining glyphs into new atlas texture if multi texture support if enabled.
            if (m_GlyphsToPack.Count > 0)
            {
                /*
                // Create new atlas texture 
                Texture2D tex = new Texture2D(m_AtlasWidth, m_AtlasHeight, TextureFormat.Alpha8, false, true);
                tex.SetPixels32(new Color32[m_AtlasWidth * m_AtlasHeight]);
                tex.Apply();

                m_AtlasTextureIndex++;

                if (m_AtlasTextures.Length == m_AtlasTextureIndex)
                    Array.Resize(ref m_AtlasTextures, Mathf.NextPowerOfTwo(m_AtlasTextureIndex + 1));

                m_AtlasTextures[m_AtlasTextureIndex] = tex;
                */
            }

            #if UNITY_EDITOR
            // Makes the changes to the font asset persistent.
            SortGlyphAndCharacterTables();
            TMP_EditorResourceManager.RegisterResourceForUpdate(this);
            #endif
        }


        internal void UpdateGlyphAdjustmentRecords(uint unicode, uint glyphIndex)
        {
            Profiler.BeginSample("TMP.UpdateGlyphAdjustmentRecords");

            int glyphCount = m_GlyphIndexList.Count;

            if (s_GlyphIndexArray.Length <= glyphCount)
                s_GlyphIndexArray = new uint[Mathf.NextPowerOfTwo(glyphCount + 1)];

            for (int i = 0; i < glyphCount; i++)
                s_GlyphIndexArray[i] = m_GlyphIndexList[i];

            // Clear unused array elements
            Array.Clear(s_GlyphIndexArray, glyphCount, s_GlyphIndexArray.Length - glyphCount);

            // Get glyph pair adjustment records from font file.
            // TODO: Revise FontEngine bindings to use a more efficient function where only the new glyph index is passed.
            GlyphPairAdjustmentRecord[] pairAdjustmentRecords = FontEngine.GetGlyphPairAdjustmentTable(s_GlyphIndexArray);

            if (pairAdjustmentRecords == null || pairAdjustmentRecords.Length == 0)
            {
                Profiler.EndSample();
                return;
            }

            if (m_FontFeatureTable == null)
                m_FontFeatureTable = new TMP_FontFeatureTable();

            for (int i = 0; i < pairAdjustmentRecords.Length && pairAdjustmentRecords[i].firstAdjustmentRecord.glyphIndex != 0; i++)
            {
                long pairKey = (long)pairAdjustmentRecords[i].secondAdjustmentRecord.glyphIndex << 32 | pairAdjustmentRecords[i].firstAdjustmentRecord.glyphIndex;

                // Check if table already contains a pair adjustment record for this key.
                if (m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.ContainsKey(pairKey))
                    continue;

                TMP_GlyphPairAdjustmentRecord record = new TMP_GlyphPairAdjustmentRecord(pairAdjustmentRecords[i]);

                m_FontFeatureTable.m_GlyphPairAdjustmentRecords.Add(record);
                m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.Add(pairKey, record);
            }

            #if UNITY_EDITOR
            m_FontFeatureTable.SortGlyphPairAdjustmentRecords();
            #endif

            Profiler.EndSample();
        }


        /// <summary>
        /// Clears font asset data including the glyph and character tables and textures.
        /// Function might be changed to Internal and only used in tests.
        /// </summary>
        /// <param name="setAtlasSizeToZero">Will set the atlas texture size to zero width and height if true.</param>
        public void ClearFontAssetData(bool setAtlasSizeToZero = false)
        {
            #if UNITY_EDITOR
            // Record full object undo in the Editor.
            //UnityEditor.Undo.RecordObjects(new UnityEngine.Object[] { this, this.atlasTexture }, "Resetting Font Asset");
            #endif

            // Clear glyph and character tables
            if (m_GlyphTable != null)
                m_GlyphTable.Clear();

            if (m_CharacterTable != null)
                m_CharacterTable.Clear();

            // Clear glyph rectangles
            if (m_UsedGlyphRects != null)
                m_UsedGlyphRects.Clear();

            if (m_FreeGlyphRects != null)
            {
                int packingModifier = ((GlyphRasterModes)m_AtlasRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP ? 0 : 1;
                m_FreeGlyphRects = new List<GlyphRect>() { new GlyphRect(0, 0, m_AtlasWidth - packingModifier, m_AtlasHeight - packingModifier) };
            }

            if (m_GlyphsToPack != null)
                m_GlyphsToPack.Clear();

            if (m_GlyphsPacked != null)
                m_GlyphsPacked.Clear();

            // Clear Glyph Adjustment Table
            if (m_FontFeatureTable != null && m_FontFeatureTable.m_GlyphPairAdjustmentRecords != null)
                m_FontFeatureTable.glyphPairAdjustmentRecords.Clear();

            m_AtlasTextureIndex = 0;

            // Clear atlas textures
            if (m_AtlasTextures != null)
            {
                for (int i = 0; i < m_AtlasTextures.Length; i++)
                {
                    Texture2D texture = m_AtlasTextures[i];

                    if (i > 0)
                        DestroyImmediate(texture, true);

                    if (texture == null)
                        continue;

                    // TODO: Need texture to be readable.
                    if (m_AtlasTextures[i].isReadable == false)
                    {
                        Debug.LogWarning("Unable to reset font asset [" + this.name + "]'s atlas texture. Please make the texture [" + m_AtlasTextures[i].name + "] readable.", m_AtlasTextures[i]);
                        continue;
                    }

                    if (setAtlasSizeToZero)
                    {
                        texture.Resize(0, 0, TextureFormat.Alpha8, false);
                    }
                    else if (texture.width != m_AtlasWidth || texture.height != m_AtlasHeight)
                    {
                        texture.Resize(m_AtlasWidth, m_AtlasHeight, TextureFormat.Alpha8, false);
                    }

                    // Clear texture atlas
                    FontEngine.ResetAtlasTexture(texture);
                    texture.Apply();

                    if (i == 0)
                        m_AtlasTexture = texture;

                    m_AtlasTextures[i] = texture;
                }
            }

            #if UNITY_EDITOR
            if (UnityEditor.EditorUtility.IsPersistent(this))
            {
                TMP_EditorResourceManager.RegisterResourceForReimport(this);
            }
            #endif

            ReadFontAssetDefinition();
        }



        /// <summary>
        /// Internal method used to upgrade font asset to support Dynamic SDF.
        /// </summary>
        private void UpgradeFontAsset()
        {
            m_Version = "1.1.0";

            Debug.Log("Upgrading font asset [" + this.name + "] to version " + m_Version + ".", this);

            m_FaceInfo.familyName = m_fontInfo.Name;
            m_FaceInfo.styleName = string.Empty;

            m_FaceInfo.pointSize = (int)m_fontInfo.PointSize;
            m_FaceInfo.scale = m_fontInfo.Scale;

            m_FaceInfo.lineHeight = m_fontInfo.LineHeight;
            m_FaceInfo.ascentLine = m_fontInfo.Ascender;
            m_FaceInfo.capLine = m_fontInfo.CapHeight;
            m_FaceInfo.meanLine = m_fontInfo.CenterLine;
            m_FaceInfo.baseline = m_fontInfo.Baseline;
            m_FaceInfo.descentLine = m_fontInfo.Descender;

            m_FaceInfo.superscriptOffset = m_fontInfo.SuperscriptOffset;
            m_FaceInfo.superscriptSize = m_fontInfo.SubSize;
            m_FaceInfo.subscriptOffset = m_fontInfo.SubscriptOffset;
            m_FaceInfo.subscriptSize = m_fontInfo.SubSize;

            m_FaceInfo.underlineOffset = m_fontInfo.Underline;
            m_FaceInfo.underlineThickness = m_fontInfo.UnderlineThickness;
            m_FaceInfo.strikethroughOffset = m_fontInfo.strikethrough;
            m_FaceInfo.strikethroughThickness = m_fontInfo.strikethroughThickness;

            m_FaceInfo.tabWidth = m_fontInfo.TabWidth;

            if (m_AtlasTextures == null || m_AtlasTextures.Length == 0)
                m_AtlasTextures = new Texture2D[1];

            m_AtlasTextures[0] = atlas;

            //atlas = null;

            m_AtlasWidth = (int)m_fontInfo.AtlasWidth;
            m_AtlasHeight = (int)m_fontInfo.AtlasHeight;
            m_AtlasPadding = (int)m_fontInfo.Padding;

            switch(m_CreationSettings.renderMode)
            {
                case 0:
                    m_AtlasRenderMode = GlyphRenderMode.SMOOTH_HINTED;
                    break;
                case 1:
                    m_AtlasRenderMode = GlyphRenderMode.SMOOTH;
                    break;
                case 2:
                    m_AtlasRenderMode = GlyphRenderMode.RASTER_HINTED;
                    break;
                case 3:
                    m_AtlasRenderMode = GlyphRenderMode.RASTER;
                    break;
                case 6:
                    m_AtlasRenderMode = GlyphRenderMode.SDF16;
                    break;
                case 7:
                    m_AtlasRenderMode = GlyphRenderMode.SDF32;
                    break;
            }

            //m_fontInfo = null;

            // Convert font weight table
            if (fontWeights != null)
            {
                m_FontWeightTable[4] = fontWeights[4];
                m_FontWeightTable[7] = fontWeights[7];

                // Clear old fontWeight
                //fontWeights = null;
            }

            // Convert font fallbacks
            if (fallbackFontAssets != null && fallbackFontAssets.Count > 0)
            {
                if (m_FallbackFontAssetTable == null)
                    m_FallbackFontAssetTable = new List<TMP_FontAsset>(fallbackFontAssets.Count);

                for (int i = 0; i < fallbackFontAssets.Count; i++)
                    m_FallbackFontAssetTable.Add(fallbackFontAssets[i]);

                // Clear old fallbackFontAssets list
                //fallbackFontAssets = null;
            }

            // Check if font asset creation settings contains a reference to the source font file GUID
            if (m_CreationSettings.sourceFontFileGUID != null || m_CreationSettings.sourceFontFileGUID != string.Empty)
            {
                m_SourceFontFileGUID = m_CreationSettings.sourceFontFileGUID;
            }
            else
            {
                Debug.LogWarning("Font asset [" + this.name + "] doesn't have a reference to its source font file. Please assign the appropriate source font file for this asset in the Font Atlas & Material section of font asset inspector.", this);
            }

            // Convert legacy glyph and character tables to new format
            m_GlyphTable.Clear();
            m_CharacterTable.Clear();

            //#if UNITY_EDITOR
            // TODO: This is causing a crash in Unity and related to AssetDatabase.LoadAssetAtPath and Resources.Load()
            // Load font to allow us to get the glyph index.
            //string path = UnityEditor.AssetDatabase.GUIDToAssetPath(m_SourceFontFileGUID);

            //if (path != string.Empty)
            //{
                //m_SourceFontFile_EditorRef = UnityEditor.AssetDatabase.LoadAssetAtPath<Font>(path);
                //FontEngine.LoadFontFace(m_SourceFontFile_EditorRef);
            //}
            //#endif

            bool isSpaceCharacterPresent = false;
            for (int i = 0; i < m_glyphInfoList.Count; i++)
            {
                TMP_Glyph oldGlyph = m_glyphInfoList[i];

                Glyph glyph = new Glyph();

                uint glyphIndex = (uint)i + 1;
                
                //#if UNITY_EDITOR
                //if (m_SourceFontFile_EditorRef != null)
                //    glyphIndex = FontEngine.GetGlyphIndex((uint)oldGlyph.id);
                //#endif

                glyph.index = glyphIndex;
                glyph.glyphRect = new GlyphRect((int)oldGlyph.x, m_AtlasHeight - (int)(oldGlyph.y + oldGlyph.height + 0.5f), (int)(oldGlyph.width + 0.5f), (int)(oldGlyph.height + 0.5f));
                glyph.metrics = new GlyphMetrics(oldGlyph.width, oldGlyph.height, oldGlyph.xOffset, oldGlyph.yOffset, oldGlyph.xAdvance);
                glyph.scale = oldGlyph.scale;
                glyph.atlasIndex = 0;

                m_GlyphTable.Add(glyph);

                TMP_Character character = new TMP_Character((uint)oldGlyph.id, glyph);

                if (oldGlyph.id == 32)
                    isSpaceCharacterPresent = true;

                m_CharacterTable.Add(character);
            }

            // Special handling for the synthesized space character
            if (!isSpaceCharacterPresent)
            {
                Debug.Log("Synthesizing Space for [" + this.name + "]");
                Glyph glyph = new Glyph(0, new GlyphMetrics(0, 0, 0, 0, m_FaceInfo.ascentLine / 5), GlyphRect.zero, 1.0f, 0);
                m_GlyphTable.Add(glyph);
                m_CharacterTable.Add(new TMP_Character(32, glyph));
            }

            // Clear legacy glyph info list.
            //m_glyphInfoList.Clear();

            ReadFontAssetDefinition();

            // Convert atlas textures data to new format
            // TODO
            #if UNITY_EDITOR
            if (UnityEditor.EditorUtility.IsPersistent(this))
            {
                TMP_EditorResourceManager.RegisterResourceForUpdate(this);
            }
            #endif
        }

        /// <summary>
        /// 
        /// </summary>
        void UpgradeGlyphAdjustmentTableToFontFeatureTable()
        {
            Debug.Log("Upgrading font asset [" + this.name + "] Glyph Adjustment Table.", this);

            if (m_FontFeatureTable == null)
                m_FontFeatureTable = new TMP_FontFeatureTable();

            int pairCount = m_KerningTable.kerningPairs.Count;

            m_FontFeatureTable.m_GlyphPairAdjustmentRecords = new List<TMP_GlyphPairAdjustmentRecord>(pairCount);

            for (int i = 0; i < pairCount; i++)
            {
                KerningPair pair = m_KerningTable.kerningPairs[i];

                uint firstGlyphIndex = 0;
                TMP_Character firstCharacter;

                if (m_CharacterLookupDictionary.TryGetValue(pair.firstGlyph, out firstCharacter))
                    firstGlyphIndex = firstCharacter.glyphIndex;

                uint secondGlyphIndex = 0;
                TMP_Character secondCharacter;

                if (m_CharacterLookupDictionary.TryGetValue(pair.secondGlyph, out secondCharacter))
                    secondGlyphIndex = secondCharacter.glyphIndex;

                TMP_GlyphAdjustmentRecord firstAdjustmentRecord = new TMP_GlyphAdjustmentRecord(firstGlyphIndex, new TMP_GlyphValueRecord(pair.firstGlyphAdjustments.xPlacement, pair.firstGlyphAdjustments.yPlacement, pair.firstGlyphAdjustments.xAdvance, pair.firstGlyphAdjustments.yAdvance));
                TMP_GlyphAdjustmentRecord secondAdjustmentRecord = new TMP_GlyphAdjustmentRecord(secondGlyphIndex, new TMP_GlyphValueRecord(pair.secondGlyphAdjustments.xPlacement, pair.secondGlyphAdjustments.yPlacement, pair.secondGlyphAdjustments.xAdvance, pair.secondGlyphAdjustments.yAdvance));
                TMP_GlyphPairAdjustmentRecord record = new TMP_GlyphPairAdjustmentRecord(firstAdjustmentRecord, secondAdjustmentRecord);

                m_FontFeatureTable.m_GlyphPairAdjustmentRecords.Add(record);
            }

            // TODO: Should clear legacy kerning table.
            m_KerningTable.kerningPairs = null;
            m_KerningTable = null;

            #if UNITY_EDITOR
            if (UnityEditor.EditorUtility.IsPersistent(this))
            {
                TMP_EditorResourceManager.RegisterResourceForUpdate(this);
            }
            #endif
        }

    }
}