using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;
using Object = UnityEngine.Object;

namespace TMPro.EditorUtilities
{
    public class TMPro_FontAssetCreatorWindow : EditorWindow
    {
        [MenuItem("Window/TextMeshPro/Font Asset Creator", false, 2025)]
        public static void ShowFontAtlasCreatorWindow()
        {
            var window = GetWindow<TMPro_FontAssetCreatorWindow>();
            window.titleContent = new GUIContent("Font Asset Creator");
            window.Focus();

            // Make sure TMP Essential Resources have been imported.
            window.CheckEssentialResources();
        }


        public static void ShowFontAtlasCreatorWindow(Font sourceFontFile)
        {
            var window = GetWindow<TMPro_FontAssetCreatorWindow>();

            window.titleContent = new GUIContent("Font Asset Creator");
            window.Focus();

            window.ClearGeneratedData();
            window.m_LegacyFontAsset = null;
            window.m_SelectedFontAsset = null;

            // Override selected font asset
            window.m_SourceFontFile = sourceFontFile;

            // Make sure TMP Essential Resources have been imported.
            window.CheckEssentialResources();
        }


        public static void ShowFontAtlasCreatorWindow(TMP_FontAsset fontAsset)
        {
            var window = GetWindow<TMPro_FontAssetCreatorWindow>();

            window.titleContent = new GUIContent("Font Asset Creator");
            window.Focus();

            // Clear any previously generated data
            window.ClearGeneratedData();
            window.m_LegacyFontAsset = null;

            // Load font asset creation settings if we have valid settings
            if (string.IsNullOrEmpty(fontAsset.creationSettings.sourceFontFileGUID) == false)
            {
                window.LoadFontCreationSettings(fontAsset.creationSettings);

                // Override settings to inject character list from font asset
                window.m_CharacterSetSelectionMode = 6;
                window.m_CharacterSequence = TMP_EditorUtility.GetUnicodeCharacterSequence(TMP_FontAsset.GetCharactersArray(fontAsset));


                window.m_ReferencedFontAsset = fontAsset;
                window.m_SavedFontAtlas = fontAsset.atlasTexture;
            }
            else
            {
                window.m_WarningMessage = "Font Asset [" + fontAsset.name + "] does not contain any previous \"Font Asset Creation Settings\". This usually means [" + fontAsset.name + "] was created before this new functionality was added.";
                window.m_SourceFontFile = null;
                window.m_LegacyFontAsset = fontAsset;
            }

            // Even if we don't have any saved generation settings, we still want to pre-select the source font file.
            window.m_SelectedFontAsset = fontAsset;

            // Make sure TMP Essential Resources have been imported.
            window.CheckEssentialResources();
        }
        
        [System.Serializable]
        class FontAssetCreationSettingsContainer
        {
            public List<FontAssetCreationSettings> fontAssetCreationSettings;
        }
        
        FontAssetCreationSettingsContainer m_FontAssetCreationSettingsContainer;
        
        //static readonly string[] m_FontCreationPresets = new string[] { "Recent 1", "Recent 2", "Recent 3", "Recent 4" };
        int m_FontAssetCreationSettingsCurrentIndex = 0;

        const string k_FontAssetCreationSettingsContainerKey = "TextMeshPro.FontAssetCreator.RecentFontAssetCreationSettings.Container";
        const string k_FontAssetCreationSettingsCurrentIndexKey = "TextMeshPro.FontAssetCreator.RecentFontAssetCreationSettings.CurrentIndex";
        const float k_TwoColumnControlsWidth = 335f;

        // Diagnostics
        System.Diagnostics.Stopwatch m_StopWatch;
        double m_GlyphPackingGenerationTime;
        double m_GlyphRenderingGenerationTime;
        
        string[] m_FontSizingOptions = { "Auto Sizing", "Custom Size" };
        int m_PointSizeSamplingMode;
        string[] m_FontResolutionLabels = { "8", "16","32", "64", "128", "256", "512", "1024", "2048", "4096", "8192" };
        int[] m_FontAtlasResolutions = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
        string[] m_FontCharacterSets = { "ASCII", "Extended ASCII", "ASCII Lowercase", "ASCII Uppercase", "Numbers + Symbols", "Custom Range", "Unicode Range (Hex)", "Custom Characters", "Characters from File" };
        enum FontPackingModes { Fast = 0, Optimum = 4 };
        FontPackingModes m_PackingMode = FontPackingModes.Fast;

        int m_CharacterSetSelectionMode;

        string m_CharacterSequence = "";
        string m_OutputFeedback = "";
        string m_WarningMessage;
        int m_CharacterCount;
        Vector2 m_ScrollPosition;
        Vector2 m_OutputScrollPosition;
        
        bool m_IsRepaintNeeded;

        float m_AtlasGenerationProgress;
        string m_AtlasGenerationProgressLabel = string.Empty;
        float m_RenderingProgress;
        bool m_IsRenderingDone;
        bool m_IsProcessing;
        bool m_IsGenerationDisabled;
        bool m_IsGenerationCancelled;

        bool m_IsFontAtlasInvalid;
        Object m_SourceFontFile;
        TMP_FontAsset m_SelectedFontAsset;
        TMP_FontAsset m_LegacyFontAsset;
        TMP_FontAsset m_ReferencedFontAsset;

        TextAsset m_CharactersFromFile;
        int m_PointSize;
        int m_Padding = 5;
        //FaceStyles m_FontStyle = FaceStyles.Normal;
        //float m_FontStyleValue = 2;

        GlyphRenderMode m_GlyphRenderMode = GlyphRenderMode.SDFAA;
        int m_AtlasWidth = 512;
        int m_AtlasHeight = 512;
        byte[] m_AtlasTextureBuffer;
        Texture2D m_FontAtlasTexture;
        Texture2D m_SavedFontAtlas;

        // 
        List<Glyph> m_FontGlyphTable = new List<Glyph>();
        List<TMP_Character> m_FontCharacterTable = new List<TMP_Character>();

        Dictionary<uint, uint> m_CharacterLookupMap = new Dictionary<uint, uint>();
        Dictionary<uint, List<uint>> m_GlyphLookupMap = new Dictionary<uint, List<uint>>();

        List<Glyph> m_GlyphsToPack = new List<Glyph>();
        List<Glyph> m_GlyphsPacked = new List<Glyph>();
        List<GlyphRect> m_FreeGlyphRects = new List<GlyphRect>();
        List<GlyphRect> m_UsedGlyphRects = new List<GlyphRect>();
        List<Glyph> m_GlyphsToRender = new List<Glyph>();
        List<uint> m_AvailableGlyphsToAdd = new List<uint>();
        List<uint> m_MissingCharacters = new List<uint>();
        List<uint> m_ExcludedCharacters = new List<uint>();

        private FaceInfo m_FaceInfo;

        bool m_IncludeFontFeatures;


        public void OnEnable()
        {
            // Used for Diagnostics
            m_StopWatch = new System.Diagnostics.Stopwatch();
            
            // Set Editor window size.
            minSize = new Vector2(315, minSize.y);

            // Initialize & Get shader property IDs.
            ShaderUtilities.GetShaderPropertyIDs();

            // Load last selected preset if we are not already in the process of regenerating an existing font asset (via the Context menu)
            if (EditorPrefs.HasKey(k_FontAssetCreationSettingsContainerKey))
            {
                if (m_FontAssetCreationSettingsContainer == null)
                    m_FontAssetCreationSettingsContainer = JsonUtility.FromJson<FontAssetCreationSettingsContainer>(EditorPrefs.GetString(k_FontAssetCreationSettingsContainerKey));

                if (m_FontAssetCreationSettingsContainer.fontAssetCreationSettings != null && m_FontAssetCreationSettingsContainer.fontAssetCreationSettings.Count > 0)
                {
                    // Load Font Asset Creation Settings preset.
                    if (EditorPrefs.HasKey(k_FontAssetCreationSettingsCurrentIndexKey))
                        m_FontAssetCreationSettingsCurrentIndex = EditorPrefs.GetInt(k_FontAssetCreationSettingsCurrentIndexKey);

                    LoadFontCreationSettings(m_FontAssetCreationSettingsContainer.fontAssetCreationSettings[m_FontAssetCreationSettingsCurrentIndex]);
                }
            }

            ClearGeneratedData();
        }


        public void OnDisable()
        {
            //Debug.Log("TextMeshPro Editor Window has been disabled.");

            // Destroy Engine only if it has been initialized already
            FontEngine.DestroyFontEngine();

            ClearGeneratedData();

            // Remove Glyph Report if one was created.
            if (File.Exists("Assets/TextMesh Pro/Glyph Report.txt"))
            {
                File.Delete("Assets/TextMesh Pro/Glyph Report.txt");
                File.Delete("Assets/TextMesh Pro/Glyph Report.txt.meta");

                AssetDatabase.Refresh();
            }

            // Save Font Asset Creation Settings Index
            SaveCreationSettingsToEditorPrefs(SaveFontCreationSettings());
            EditorPrefs.SetInt(k_FontAssetCreationSettingsCurrentIndexKey, m_FontAssetCreationSettingsCurrentIndex);

            // Unregister to event
            TMPro_EventManager.RESOURCE_LOAD_EVENT.Remove(ON_RESOURCES_LOADED);

            Resources.UnloadUnusedAssets();
        }


        // Event received when TMP resources have been loaded.
        void ON_RESOURCES_LOADED()
        {
            TMPro_EventManager.RESOURCE_LOAD_EVENT.Remove(ON_RESOURCES_LOADED);

            m_IsGenerationDisabled = false;
        }

        // Make sure TMP Essential Resources have been imported.
        void CheckEssentialResources()
        {
            if (TMP_Settings.instance == null)
            {
                if (m_IsGenerationDisabled == false)
                    TMPro_EventManager.RESOURCE_LOAD_EVENT.Add(ON_RESOURCES_LOADED);

                m_IsGenerationDisabled = true;
            }
        }


        public void OnGUI()
        {
            GUILayout.BeginHorizontal();
            DrawControls();
            if (position.width > position.height && position.width > k_TwoColumnControlsWidth)
            {
                DrawPreview();
            }
            GUILayout.EndHorizontal();
        }


        public void Update()
        {
            if (m_IsRepaintNeeded)
            {
                //Debug.Log("Repainting...");
                m_IsRepaintNeeded = false;
                Repaint();
            }

            // Update Progress bar is we are Rendering a Font.
            if (m_IsProcessing)
            {
                m_AtlasGenerationProgress = FontEngine.generationProgress;

                m_IsRepaintNeeded = true;
            }

            // Update Feedback Window & Create Font Texture once Rendering is done.
            if (m_IsRenderingDone)
            {
                m_IsProcessing = false;
                m_IsRenderingDone = false;

                if (m_IsGenerationCancelled == false)
                {
                    m_AtlasGenerationProgressLabel = "Generation completed in: " + (m_GlyphPackingGenerationTime + m_GlyphRenderingGenerationTime).ToString("0.00 ms.");

                    UpdateRenderFeedbackWindow();
                    CreateFontAtlasTexture();

                    // If dynamic make readable ...
                    m_FontAtlasTexture.Apply(false, false);
                }
                Repaint();
            }
        }


        /// <summary>
        /// Method which returns the character corresponding to a decimal value.
        /// </summary>
        /// <param name="sequence"></param>
        /// <returns></returns>
        static uint[] ParseNumberSequence(string sequence)
        {
            List<uint> unicodeList = new List<uint>();
            string[] sequences = sequence.Split(',');

            foreach (string seq in sequences)
            {
                string[] s1 = seq.Split('-');

                if (s1.Length == 1)
                    try
                    {
                        unicodeList.Add(uint.Parse(s1[0]));
                    }
                    catch
                    {
                        Debug.Log("No characters selected or invalid format.");
                    }
                else
                {
                    for (uint j = uint.Parse(s1[0]); j < uint.Parse(s1[1]) + 1; j++)
                    {
                        unicodeList.Add(j);
                    }
                }
            }

            return unicodeList.ToArray();
        }


        /// <summary>
        /// Method which returns the character (decimal value) from a hex sequence.
        /// </summary>
        /// <param name="sequence"></param>
        /// <returns></returns>
        static uint[] ParseHexNumberSequence(string sequence)
        {
            List<uint> unicodeList = new List<uint>();
            string[] sequences = sequence.Split(',');

            foreach (string seq in sequences)
            {
                string[] s1 = seq.Split('-');

                if (s1.Length == 1)
                    try
                    {
                        unicodeList.Add(uint.Parse(s1[0], NumberStyles.AllowHexSpecifier));
                    }
                    catch
                    {
                        Debug.Log("No characters selected or invalid format.");
                    }
                else
                {
                    for (uint j = uint.Parse(s1[0], NumberStyles.AllowHexSpecifier); j < uint.Parse(s1[1], NumberStyles.AllowHexSpecifier) + 1; j++)
                    {
                        unicodeList.Add(j);
                    }
                }
            }

            return unicodeList.ToArray();
        }


        void DrawControls()
        {
            GUILayout.Space(5f);

            if (position.width > position.height && position.width > k_TwoColumnControlsWidth)
            {
                m_ScrollPosition = EditorGUILayout.BeginScrollView(m_ScrollPosition, GUILayout.Width(315));
            }
            else
            {
                m_ScrollPosition = EditorGUILayout.BeginScrollView(m_ScrollPosition);
            }
            
            GUILayout.Space(5f);

            GUILayout.Label(m_SelectedFontAsset != null ? string.Format("Font Settings [{0}]", m_SelectedFontAsset.name) : "Font Settings", EditorStyles.boldLabel);

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            EditorGUIUtility.labelWidth = 125f;
            EditorGUIUtility.fieldWidth = 5f;
            
            // Disable Options if already generating a font atlas texture.
            EditorGUI.BeginDisabledGroup(m_IsProcessing);
            {
                // FONT TTF SELECTION
                EditorGUI.BeginChangeCheck();
                m_SourceFontFile = EditorGUILayout.ObjectField("Source Font File", m_SourceFontFile, typeof(Font), false) as Font;
                if (EditorGUI.EndChangeCheck())
                {
                    m_SelectedFontAsset = null;
                    m_IsFontAtlasInvalid = true;
                }

                // FONT SIZING
                EditorGUI.BeginChangeCheck();
                if (m_PointSizeSamplingMode == 0)
                {
                    m_PointSizeSamplingMode = EditorGUILayout.Popup("Sampling Point Size", m_PointSizeSamplingMode, m_FontSizingOptions);
                }
                else
                {
                    GUILayout.BeginHorizontal();
                    m_PointSizeSamplingMode = EditorGUILayout.Popup("Sampling Point Size", m_PointSizeSamplingMode, m_FontSizingOptions, GUILayout.Width(225));
                    m_PointSize = EditorGUILayout.IntField(m_PointSize);
                    GUILayout.EndHorizontal();
                }
                if (EditorGUI.EndChangeCheck())
                {
                    m_IsFontAtlasInvalid = true;
                }

                // FONT PADDING
                EditorGUI.BeginChangeCheck();
                m_Padding = EditorGUILayout.IntField("Padding", m_Padding);
                m_Padding = (int)Mathf.Clamp(m_Padding, 0f, 64f);
                if (EditorGUI.EndChangeCheck())
                {
                    m_IsFontAtlasInvalid = true;
                }

                // FONT PACKING METHOD SELECTION
                EditorGUI.BeginChangeCheck();
                m_PackingMode = (FontPackingModes)EditorGUILayout.EnumPopup("Packing Method", m_PackingMode);
                if (EditorGUI.EndChangeCheck())
                {
                    m_IsFontAtlasInvalid = true;
                }

                // FONT ATLAS RESOLUTION SELECTION
                GUILayout.BeginHorizontal();
                GUI.changed = false;

                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PrefixLabel("Atlas Resolution");
                m_AtlasWidth = EditorGUILayout.IntPopup(m_AtlasWidth, m_FontResolutionLabels, m_FontAtlasResolutions);
                m_AtlasHeight = EditorGUILayout.IntPopup(m_AtlasHeight, m_FontResolutionLabels, m_FontAtlasResolutions);
                if (EditorGUI.EndChangeCheck())
                {
                    m_IsFontAtlasInvalid = true;
                }

                GUILayout.EndHorizontal();


                // FONT CHARACTER SET SELECTION
                EditorGUI.BeginChangeCheck();
                bool hasSelectionChanged = false;
                m_CharacterSetSelectionMode = EditorGUILayout.Popup("Character Set", m_CharacterSetSelectionMode, m_FontCharacterSets);
                if (EditorGUI.EndChangeCheck())
                {
                    m_CharacterSequence = "";
                    hasSelectionChanged = true;
                    m_IsFontAtlasInvalid = true;
                }

                switch (m_CharacterSetSelectionMode)
                {
                    case 0: // ASCII
                        //characterSequence = "32 - 126, 130, 132 - 135, 139, 145 - 151, 153, 155, 161, 166 - 167, 169 - 174, 176, 181 - 183, 186 - 187, 191, 8210 - 8226, 8230, 8240, 8242 - 8244, 8249 - 8250, 8252 - 8254, 8260, 8286";
                        m_CharacterSequence = "32 - 126, 160, 8203, 8230, 9633";
                        break;

                    case 1: // EXTENDED ASCII
                        m_CharacterSequence = "32 - 126, 160 - 255, 8192 - 8303, 8364, 8482, 9633";
                        // Could add 9632 for missing glyph
                        break;

                    case 2: // Lowercase
                        m_CharacterSequence = "32 - 64, 91 - 126, 160";
                        break;

                    case 3: // Uppercase
                        m_CharacterSequence = "32 - 96, 123 - 126, 160";
                        break;

                    case 4: // Numbers & Symbols
                        m_CharacterSequence = "32 - 64, 91 - 96, 123 - 126, 160";
                        break;

                    case 5: // Custom Range
                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                        GUILayout.Label("Enter a sequence of decimal values to define the characters to be included in the font asset or retrieve one from another font asset.", TMP_UIStyleManager.label);
                        GUILayout.Space(10f);

                        EditorGUI.BeginChangeCheck();
                        m_ReferencedFontAsset = EditorGUILayout.ObjectField("Select Font Asset", m_ReferencedFontAsset, typeof(TMP_FontAsset), false) as TMP_FontAsset;
                        if (EditorGUI.EndChangeCheck() || hasSelectionChanged)
                        {
                            if (m_ReferencedFontAsset != null)
                                m_CharacterSequence = TMP_EditorUtility.GetDecimalCharacterSequence(TMP_FontAsset.GetCharactersArray(m_ReferencedFontAsset));
                            
                            m_IsFontAtlasInvalid = true;
                        }

                        // Filter out unwanted characters.
                        char chr = Event.current.character;
                        if ((chr < '0' || chr > '9') && (chr < ',' || chr > '-'))
                        {
                            Event.current.character = '\0';
                        }
                        GUILayout.Label("Character Sequence (Decimal)", EditorStyles.boldLabel);
                        EditorGUI.BeginChangeCheck();
                        m_CharacterSequence = EditorGUILayout.TextArea(m_CharacterSequence, TMP_UIStyleManager.textAreaBoxWindow, GUILayout.Height(120), GUILayout.ExpandWidth(true));
                        if (EditorGUI.EndChangeCheck())
                        {
                            m_IsFontAtlasInvalid = true;
                        }
                        
                        EditorGUILayout.EndVertical();
                        break;

                    case 6: // Unicode HEX Range
                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                        GUILayout.Label("Enter a sequence of Unicode (hex) values to define the characters to be included in the font asset or retrieve one from another font asset.", TMP_UIStyleManager.label);
                        GUILayout.Space(10f);

                        EditorGUI.BeginChangeCheck();
                        m_ReferencedFontAsset = EditorGUILayout.ObjectField("Select Font Asset", m_ReferencedFontAsset, typeof(TMP_FontAsset), false) as TMP_FontAsset;
                        if (EditorGUI.EndChangeCheck() || hasSelectionChanged)
                        {
                            if (m_ReferencedFontAsset != null)
                                m_CharacterSequence = TMP_EditorUtility.GetUnicodeCharacterSequence(TMP_FontAsset.GetCharactersArray(m_ReferencedFontAsset));
                            
                            m_IsFontAtlasInvalid = true;
                        }

                        // Filter out unwanted characters.
                        chr = Event.current.character;
                        if ((chr < '0' || chr > '9') && (chr < 'a' || chr > 'f') && (chr < 'A' || chr > 'F') && (chr < ',' || chr > '-'))
                        {
                            Event.current.character = '\0';
                        }
                        GUILayout.Label("Character Sequence (Hex)", EditorStyles.boldLabel);
                        EditorGUI.BeginChangeCheck();
                        m_CharacterSequence = EditorGUILayout.TextArea(m_CharacterSequence, TMP_UIStyleManager.textAreaBoxWindow, GUILayout.Height(120), GUILayout.ExpandWidth(true));
                        if (EditorGUI.EndChangeCheck())
                        {
                            m_IsFontAtlasInvalid = true;
                        }

                        EditorGUILayout.EndVertical();
                        break;

                    case 7: // Characters from Font Asset
                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                        GUILayout.Label("Type the characters to be included in the font asset or retrieve them from another font asset.", TMP_UIStyleManager.label);
                        GUILayout.Space(10f);

                        EditorGUI.BeginChangeCheck();
                        m_ReferencedFontAsset = EditorGUILayout.ObjectField("Select Font Asset", m_ReferencedFontAsset, typeof(TMP_FontAsset), false) as TMP_FontAsset;
                        if (EditorGUI.EndChangeCheck() || hasSelectionChanged)
                        {
                            if (m_ReferencedFontAsset != null)
                                m_CharacterSequence = TMP_FontAsset.GetCharacters(m_ReferencedFontAsset);
                            
                            m_IsFontAtlasInvalid = true;
                        }

                        EditorGUI.indentLevel = 0;
                        
                        GUILayout.Label("Custom Character List", EditorStyles.boldLabel);
                        EditorGUI.BeginChangeCheck();
                        m_CharacterSequence = EditorGUILayout.TextArea(m_CharacterSequence, TMP_UIStyleManager.textAreaBoxWindow, GUILayout.Height(120), GUILayout.ExpandWidth(true));
                        if (EditorGUI.EndChangeCheck())
                        {
                            m_IsFontAtlasInvalid = true;
                        }
                        EditorGUILayout.EndVertical();
                        break;

                    case 8: // Character List from File
                        EditorGUI.BeginChangeCheck();
                        m_CharactersFromFile = EditorGUILayout.ObjectField("Character File", m_CharactersFromFile, typeof(TextAsset), false) as TextAsset;
                        if (EditorGUI.EndChangeCheck())
                        {
                            m_IsFontAtlasInvalid = true;
                        }

                        if (m_CharactersFromFile != null)
                        {
                            Regex rx = new Regex(@"(?<!\\)(?:\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})");

                            m_CharacterSequence = rx.Replace(m_CharactersFromFile.text,
                                match =>
                                {
                                    if (match.Value.StartsWith("\\U"))
                                        return char.ConvertFromUtf32(int.Parse(match.Value.Replace("\\U", ""), NumberStyles.HexNumber));

                                    return char.ConvertFromUtf32(int.Parse(match.Value.Replace("\\u", ""), NumberStyles.HexNumber));
                                });
                        }
                        break;
                }

                // FONT STYLE SELECTION
                //GUILayout.BeginHorizontal();
                //EditorGUI.BeginChangeCheck();
                ////m_FontStyle = (FaceStyles)EditorGUILayout.EnumPopup("Font Style", m_FontStyle, GUILayout.Width(225));
                ////m_FontStyleValue = EditorGUILayout.IntField((int)m_FontStyleValue);
                //if (EditorGUI.EndChangeCheck())
                //{
                //    m_IsFontAtlasInvalid = true;
                //}
                //GUILayout.EndHorizontal();

                // Render Mode Selection
                CheckForLegacyGlyphRenderMode();

                EditorGUI.BeginChangeCheck();
                m_GlyphRenderMode = (GlyphRenderMode)EditorGUILayout.EnumPopup("Render Mode", m_GlyphRenderMode);
                if (EditorGUI.EndChangeCheck())
                {
                    m_IsFontAtlasInvalid = true;
                }

                m_IncludeFontFeatures = EditorGUILayout.Toggle("Get Kerning Pairs", m_IncludeFontFeatures);

                EditorGUILayout.Space();
            }

            EditorGUI.EndDisabledGroup();

            if (!string.IsNullOrEmpty(m_WarningMessage))
            {
                EditorGUILayout.HelpBox(m_WarningMessage, MessageType.Warning);
            }
            
            GUI.enabled = m_SourceFontFile != null && !m_IsProcessing && !m_IsGenerationDisabled; // Enable Preview if we are not already rendering a font.
            if (GUILayout.Button("Generate Font Atlas") && GUI.enabled)
            {
                if (!m_IsProcessing && m_SourceFontFile != null)
                {
                    DestroyImmediate(m_FontAtlasTexture);
                    m_FontAtlasTexture = null;
                    m_SavedFontAtlas = null;

                    // Initialize font engine
                    FontEngineError errorCode = FontEngine.InitializeFontEngine();
                    if (errorCode != FontEngineError.Success)
                    {
                        Debug.Log("Font Asset Creator - Error [" + errorCode + "] has occurred while Initializing the FreeType Library.");
                    }
                    
                    // Get file path of the source font file.
                    string fontPath = AssetDatabase.GetAssetPath(m_SourceFontFile);

                    if (errorCode == FontEngineError.Success)
                    {
                        errorCode = FontEngine.LoadFontFace(fontPath);

                        if (errorCode != FontEngineError.Success)
                        {
                            Debug.Log("Font Asset Creator - Error Code [" + errorCode + "] has occurred trying to load the [" + m_SourceFontFile.name + "] font file. This typically results from the use of an incompatible or corrupted font file.");
                        }
                    }


                    // Define an array containing the characters we will render.
                    if (errorCode == FontEngineError.Success)
                    {
                        uint[] characterSet = null;

                        // Get list of characters that need to be packed and rendered to the atlas texture.
                        if (m_CharacterSetSelectionMode == 7 || m_CharacterSetSelectionMode == 8)
                        {
                            List<uint> char_List = new List<uint>();

                            for (int i = 0; i < m_CharacterSequence.Length; i++)
                            {
                                uint unicode = m_CharacterSequence[i];

                                // Handle surrogate pairs
                                if (i < m_CharacterSequence.Length - 1 && char.IsHighSurrogate((char)unicode) && char.IsLowSurrogate(m_CharacterSequence[i + 1]))
                                {
                                    unicode = (uint)char.ConvertToUtf32(m_CharacterSequence[i], m_CharacterSequence[i + 1]);
                                    i += 1;
                                }

                                // Check to make sure we don't include duplicates
                                if (char_List.FindIndex(item => item == unicode) == -1)
                                    char_List.Add(unicode);
                            }

                            characterSet = char_List.ToArray();
                        }
                        else if (m_CharacterSetSelectionMode == 6)
                        {
                            characterSet = ParseHexNumberSequence(m_CharacterSequence);
                        }
                        else
                        {
                            characterSet = ParseNumberSequence(m_CharacterSequence);
                        }

                        m_CharacterCount = characterSet.Length;
                        
                        m_AtlasGenerationProgress = 0;
                        m_IsProcessing = true;
                        m_IsGenerationCancelled = false;

                        GlyphLoadFlags glyphLoadFlags = ((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_HINTED) == GlyphRasterModes.RASTER_MODE_HINTED ? GlyphLoadFlags.LOAD_RENDER : GlyphLoadFlags.LOAD_RENDER | GlyphLoadFlags.LOAD_NO_HINTING;

                        // 
                        AutoResetEvent autoEvent = new AutoResetEvent(false);

                        // Worker thread to pack glyphs in the given texture space.
                        ThreadPool.QueueUserWorkItem(PackGlyphs =>
                        {
                            // Start Stop Watch
                            m_StopWatch = System.Diagnostics.Stopwatch.StartNew();

                            // Clear the various lists used in the generation process.
                            m_AvailableGlyphsToAdd.Clear();
                            m_MissingCharacters.Clear();
                            m_ExcludedCharacters.Clear();
                            m_CharacterLookupMap.Clear();
                            m_GlyphLookupMap.Clear();
                            m_GlyphsToPack.Clear();
                            m_GlyphsPacked.Clear();

                            // Check if requested characters are available in the source font file.
                            for (int i = 0; i < characterSet.Length; i++)
                            {
                                uint unicode = characterSet[i];
                                uint glyphIndex;

                                if (FontEngine.TryGetGlyphIndex(unicode, out glyphIndex))
                                {
                                    // Skip over potential duplicate characters.
                                    if (m_CharacterLookupMap.ContainsKey(unicode))
                                        continue;

                                    // Add character to character lookup map.
                                    m_CharacterLookupMap.Add(unicode, glyphIndex);

                                    // Skip over potential duplicate glyph references.
                                    if (m_GlyphLookupMap.ContainsKey(glyphIndex))
                                    {
                                        // Add additional glyph reference for this character.
                                        m_GlyphLookupMap[glyphIndex].Add(unicode);
                                        continue;
                                    }

                                    // Add glyph reference to glyph lookup map.
                                    m_GlyphLookupMap.Add(glyphIndex, new List<uint>() { unicode });

                                    // Add glyph index to list of glyphs to add to texture.
                                    m_AvailableGlyphsToAdd.Add(glyphIndex);
                                }
                                else
                                {
                                    // Add Unicode to list of missing characters.
                                    m_MissingCharacters.Add(unicode);
                                }
                            }

                            // Pack available glyphs in the provided texture space.
                            if (m_AvailableGlyphsToAdd.Count > 0)
                            {
                                int packingModifier = ((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP ? 0 : 1;

                                if (m_PointSizeSamplingMode == 0) // Auto-Sizing Point Size Mode
                                {
                                    // Estimate min / max range for auto sizing of point size.
                                    int minPointSize = 0;
                                    int maxPointSize = (int)Mathf.Sqrt((m_AtlasWidth * m_AtlasHeight) / m_AvailableGlyphsToAdd.Count) * 3;

                                    m_PointSize = (maxPointSize + minPointSize) / 2;

                                    bool optimumPointSizeFound = false;
                                    for (int iteration = 0; iteration < 15 && optimumPointSizeFound == false; iteration++)
                                    {
                                        m_AtlasGenerationProgressLabel = "Packing glyphs - Pass (" + iteration + ")";

                                        FontEngine.SetFaceSize(m_PointSize);

                                        m_GlyphsToPack.Clear();
                                        m_GlyphsPacked.Clear();

                                        m_FreeGlyphRects.Clear();
                                        m_FreeGlyphRects.Add(new GlyphRect(0, 0, m_AtlasWidth - packingModifier, m_AtlasHeight - packingModifier));
                                        m_UsedGlyphRects.Clear();

                                        for (int i = 0; i < m_AvailableGlyphsToAdd.Count; i++)
                                        {
                                            uint glyphIndex = m_AvailableGlyphsToAdd[i];
                                            Glyph glyph;

                                            if (FontEngine.TryGetGlyphWithIndexValue(glyphIndex, glyphLoadFlags, out glyph))
                                            {
                                                if (glyph.glyphRect.width > 0 && glyph.glyphRect.height > 0)
                                                {
                                                    m_GlyphsToPack.Add(glyph);
                                                }
                                                else
                                                {
                                                    m_GlyphsPacked.Add(glyph);
                                                }
                                            }
                                        }

                                        FontEngine.TryPackGlyphsInAtlas(m_GlyphsToPack, m_GlyphsPacked, m_Padding, (GlyphPackingMode)m_PackingMode, m_GlyphRenderMode, m_AtlasWidth, m_AtlasHeight, m_FreeGlyphRects, m_UsedGlyphRects);

                                        if (m_IsGenerationCancelled)
                                        {
                                            DestroyImmediate(m_FontAtlasTexture);
                                            m_FontAtlasTexture = null;
                                            return;
                                        }

                                        //Debug.Log("Glyphs remaining to add [" + m_GlyphsToAdd.Count + "]. Glyphs added [" + m_GlyphsAdded.Count + "].");

                                        if (m_GlyphsToPack.Count > 0)
                                        {
                                            if (m_PointSize > minPointSize)
                                            {
                                                maxPointSize = m_PointSize;
                                                m_PointSize = (m_PointSize + minPointSize) / 2;

                                                //Debug.Log("Decreasing point size from [" + maxPointSize + "] to [" + m_PointSize + "].");
                                            }
                                        }
                                        else
                                        {
                                            if (maxPointSize - minPointSize > 1 && m_PointSize < maxPointSize)
                                            {
                                                minPointSize = m_PointSize;
                                                m_PointSize = (m_PointSize + maxPointSize) / 2;

                                                //Debug.Log("Increasing point size from [" + minPointSize + "] to [" + m_PointSize + "].");
                                            }
                                            else
                                            {
                                                //Debug.Log("[" + iteration + "] iterations to find the optimum point size of : [" + m_PointSize + "].");
                                                optimumPointSizeFound = true;
                                            }
                                        }
                                    }
                                }
                                else // Custom Point Size Mode
                                {
                                    m_AtlasGenerationProgressLabel = "Packing glyphs...";

                                    // Set point size
                                    FontEngine.SetFaceSize(m_PointSize);

                                    m_GlyphsToPack.Clear();
                                    m_GlyphsPacked.Clear();

                                    m_FreeGlyphRects.Clear();
                                    m_FreeGlyphRects.Add(new GlyphRect(0, 0, m_AtlasWidth - packingModifier, m_AtlasHeight - packingModifier));
                                    m_UsedGlyphRects.Clear();

                                    for (int i = 0; i < m_AvailableGlyphsToAdd.Count; i++)
                                    {
                                        uint glyphIndex = m_AvailableGlyphsToAdd[i];
                                        Glyph glyph;

                                        if (FontEngine.TryGetGlyphWithIndexValue(glyphIndex, glyphLoadFlags, out glyph))
                                        {
                                            if (glyph.glyphRect.width > 0 && glyph.glyphRect.height > 0)
                                            {
                                                m_GlyphsToPack.Add(glyph);
                                            }
                                            else
                                            {
                                                m_GlyphsPacked.Add(glyph);
                                            }
                                        }
                                    }

                                    FontEngine.TryPackGlyphsInAtlas(m_GlyphsToPack, m_GlyphsPacked, m_Padding, (GlyphPackingMode)m_PackingMode, m_GlyphRenderMode, m_AtlasWidth, m_AtlasHeight, m_FreeGlyphRects, m_UsedGlyphRects);

                                    if (m_IsGenerationCancelled)
                                    {
                                        DestroyImmediate(m_FontAtlasTexture);
                                        m_FontAtlasTexture = null;
                                        return;
                                    }
                                    //Debug.Log("Glyphs remaining to add [" + m_GlyphsToAdd.Count + "]. Glyphs added [" + m_GlyphsAdded.Count + "].");
                                }

                            }
                            else
                            {
                                int packingModifier = ((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP ? 0 : 1;

                                FontEngine.SetFaceSize(m_PointSize);

                                m_GlyphsToPack.Clear();
                                m_GlyphsPacked.Clear();

                                m_FreeGlyphRects.Clear();
                                m_FreeGlyphRects.Add(new GlyphRect(0, 0, m_AtlasWidth - packingModifier, m_AtlasHeight - packingModifier));
                                m_UsedGlyphRects.Clear();
                            }

                            //Stop StopWatch
                            m_StopWatch.Stop();
                            m_GlyphPackingGenerationTime = m_StopWatch.Elapsed.TotalMilliseconds;
                            Debug.Log("Glyph packing completed in: " + m_GlyphPackingGenerationTime.ToString("0.000 ms."));
                            m_StopWatch.Reset();

                            m_FontCharacterTable.Clear();
                            m_FontGlyphTable.Clear();
                            m_GlyphsToRender.Clear();

                            // Add glyphs and characters successfully added to texture to their respective font tables.
                            foreach (Glyph glyph in m_GlyphsPacked)
                            {
                                uint glyphIndex = glyph.index;

                                m_FontGlyphTable.Add(glyph);

                                // Add glyphs to list of glyphs that need to be rendered.
                                if (glyph.glyphRect.width > 0 && glyph.glyphRect.height > 0)
                                    m_GlyphsToRender.Add(glyph);
                        
                                foreach (uint unicode in m_GlyphLookupMap[glyphIndex])
                                {
                                    // Create new Character
                                    m_FontCharacterTable.Add(new TMP_Character(unicode, glyph));
                                }
                            }

                            // 
                            foreach (Glyph glyph in m_GlyphsToPack)
                            {
                                foreach (uint unicode in m_GlyphLookupMap[glyph.index])
                                {
                                    m_ExcludedCharacters.Add(unicode);
                                }
                            }

                            // Get the face info for the current sampling point size.
                            m_FaceInfo = FontEngine.GetFaceInfo();

                            autoEvent.Set();
                        });

                        // Worker thread to render glyphs in texture buffer.
                        ThreadPool.QueueUserWorkItem(RenderGlyphs =>
                        {
                            autoEvent.WaitOne();

                            // Start Stop Watch
                            m_StopWatch = System.Diagnostics.Stopwatch.StartNew();

                            m_IsRenderingDone = false;

                            // Allocate texture data
                            m_AtlasTextureBuffer = new byte[m_AtlasWidth * m_AtlasHeight];

                            m_AtlasGenerationProgressLabel = "Rendering glyphs...";

                            // Render and add glyphs to the given atlas texture.
                            if (m_GlyphsToRender.Count > 0)
                            {
                                FontEngine.RenderGlyphsToTexture(m_GlyphsToRender, m_Padding, m_GlyphRenderMode, m_AtlasTextureBuffer, m_AtlasWidth, m_AtlasHeight);
                            }

                            m_IsRenderingDone = true;

                            // Stop StopWatch
                            m_StopWatch.Stop();
                            m_GlyphRenderingGenerationTime = m_StopWatch.Elapsed.TotalMilliseconds;
                            Debug.Log("Font Atlas generation completed in: " + m_GlyphRenderingGenerationTime.ToString("0.000 ms."));
                            m_StopWatch.Reset();
                        });
                    }

                    SaveCreationSettingsToEditorPrefs(SaveFontCreationSettings());
                }
            }

            // FONT RENDERING PROGRESS BAR
            GUILayout.Space(1);
            Rect progressRect = EditorGUILayout.GetControlRect(false, 20);

            GUI.enabled = true;
            progressRect.width -= 22;
            EditorGUI.ProgressBar(progressRect, Mathf.Max(0.01f, m_AtlasGenerationProgress), m_AtlasGenerationProgressLabel);
            progressRect.x = progressRect.x + progressRect.width + 2;
            progressRect.y -= 1;
            progressRect.width = 20;
            progressRect.height = 20;

            GUI.enabled = m_IsProcessing;
            if (GUI.Button(progressRect, "X"))
            {
                FontEngine.SendCancellationRequest();
                m_AtlasGenerationProgress = 0;
                m_IsProcessing = false;
                m_IsGenerationCancelled = true;
            }
            GUILayout.Space(5);

            // FONT STATUS & INFORMATION
            GUI.enabled = true;
            
            GUILayout.BeginVertical(EditorStyles.helpBox, GUILayout.Height(200));
            m_OutputScrollPosition = EditorGUILayout.BeginScrollView(m_OutputScrollPosition);
            EditorGUILayout.LabelField(m_OutputFeedback, TMP_UIStyleManager.label);
            EditorGUILayout.EndScrollView();
            GUILayout.EndVertical();

            // SAVE TEXTURE & CREATE and SAVE FONT XML FILE
            GUI.enabled = m_FontAtlasTexture != null && !m_IsProcessing;    // Enable Save Button if font_Atlas is not Null.
            
            EditorGUILayout.BeginHorizontal();
                
            if (GUILayout.Button("Save") && GUI.enabled)
            {
                if (m_SelectedFontAsset == null)
                {
                    if (m_LegacyFontAsset != null)
                        SaveNewFontAssetWithSameName(m_LegacyFontAsset);
                    else
                        SaveNewFontAsset(m_SourceFontFile);
                }
                else
                {
                    // Save over exiting Font Asset
                    string filePath = Path.GetFullPath(AssetDatabase.GetAssetPath(m_SelectedFontAsset)).Replace('\\', '/');

                    if (((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP)
                            Save_Bitmap_FontAsset(filePath);
                        else
                        Save_SDF_FontAsset(filePath);
                }
            }
            if (GUILayout.Button("Save as...") && GUI.enabled)
            {
                if (m_SelectedFontAsset == null)
                {
                    SaveNewFontAsset(m_SourceFontFile);
                }
                else
                {
                    SaveNewFontAssetWithSameName(m_SelectedFontAsset);
                }
            }
                
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space();

            EditorGUILayout.EndVertical();
            
            GUI.enabled = true; // Re-enable GUI

            if (position.height > position.width || position.width < k_TwoColumnControlsWidth)
            {
                DrawPreview();
                GUILayout.Space(5);
            }
            
            EditorGUILayout.EndScrollView();

            if (m_IsFontAtlasInvalid)
                ClearGeneratedData();
        }


        /// <summary>
        /// Clear the previously generated data.
        /// </summary>
        void ClearGeneratedData()
        {
            m_IsFontAtlasInvalid = false;

            if (m_FontAtlasTexture != null && !EditorUtility.IsPersistent(m_FontAtlasTexture))
            {
                DestroyImmediate(m_FontAtlasTexture);
                m_FontAtlasTexture = null;
            }
        
            m_AtlasGenerationProgressLabel = string.Empty;
            m_AtlasGenerationProgress = 0;
            m_SavedFontAtlas = null;

            m_OutputFeedback = string.Empty;
            m_WarningMessage = string.Empty;
        }


        /// <summary>
        /// Function to update the feedback window showing the results of the latest generation.
        /// </summary>
        void UpdateRenderFeedbackWindow()
        {
            m_PointSize = m_FaceInfo.pointSize;

            string missingGlyphReport = string.Empty;

            //string colorTag = m_FontCharacterTable.Count == m_CharacterCount ? "<color=#C0ffff>" : "<color=#ffff00>";
            string colorTag2 = "<color=#C0ffff>";

            missingGlyphReport = "Font: <b>" + colorTag2 + m_FaceInfo.familyName + "</color></b>  Style: <b>" + colorTag2 + m_FaceInfo.styleName + "</color></b>";

            missingGlyphReport += "\nPoint Size: <b>" + colorTag2 + m_FaceInfo.pointSize + "</color></b>   SP/PD Ratio: <b>" + colorTag2 +  ((float)m_Padding / m_FaceInfo.pointSize).ToString("0.0%" + "</color></b>");

            missingGlyphReport += "\n\nCharacters included: <color=#ffff00><b>" + m_FontCharacterTable.Count + "/" + m_CharacterCount + "</b></color>";
            missingGlyphReport += "\nMissing characters: <color=#ffff00><b>" + m_MissingCharacters.Count + "</b></color>";
            missingGlyphReport += "\nExcluded characters: <color=#ffff00><b>" + m_ExcludedCharacters.Count + "</b></color>";

            // Report characters missing from font file
            missingGlyphReport += "\n\n<b><color=#ffff00>Characters missing from font file:</color></b>";
            missingGlyphReport += "\n----------------------------------------";
            
            m_OutputFeedback = missingGlyphReport;

            for (int i = 0; i < m_MissingCharacters.Count; i++)
            {
                missingGlyphReport += "\nID: <color=#C0ffff>" + m_MissingCharacters[i] + "\t</color>Hex: <color=#C0ffff>" + m_MissingCharacters[i].ToString("X") + "\t</color>Char [<color=#C0ffff>" + (char)m_MissingCharacters[i] + "</color>]";

                if (missingGlyphReport.Length < 16300)
                    m_OutputFeedback = missingGlyphReport;
            }

            // Report characters that did not fit in the atlas texture
            missingGlyphReport += "\n\n<b><color=#ffff00>Characters excluded from packing:</color></b>";
            missingGlyphReport += "\n----------------------------------------";

            for (int i = 0; i < m_ExcludedCharacters.Count; i++)
                {
                missingGlyphReport += "\nID: <color=#C0ffff>" + m_ExcludedCharacters[i] + "\t</color>Hex: <color=#C0ffff>" + m_ExcludedCharacters[i].ToString("X") + "\t</color>Char [<color=#C0ffff>" + (char)m_ExcludedCharacters[i] + "</color>]";

                    if (missingGlyphReport.Length < 16300)
                    m_OutputFeedback = missingGlyphReport;
            }

            if (missingGlyphReport.Length > 16300)
                m_OutputFeedback += "\n\n<color=#ffff00>Report truncated.</color>\n<color=#c0ffff>See</color> \"TextMesh Pro\\Glyph Report.txt\"";

            // Save Missing Glyph Report file
            if (Directory.Exists("Assets/TextMesh Pro"))
            {
                missingGlyphReport = System.Text.RegularExpressions.Regex.Replace(missingGlyphReport, @"<[^>]*>", string.Empty);
                File.WriteAllText("Assets/TextMesh Pro/Glyph Report.txt", missingGlyphReport);
                AssetDatabase.Refresh();
            }
        }


        void CreateFontAtlasTexture()
        {
            if (m_FontAtlasTexture != null)
                DestroyImmediate(m_FontAtlasTexture);

            m_FontAtlasTexture = new Texture2D(m_AtlasWidth, m_AtlasHeight, TextureFormat.Alpha8, false, true);

            Color32[] colors = new Color32[m_AtlasWidth * m_AtlasHeight];

            for (int i = 0; i < colors.Length; i++)
            {
                byte c = m_AtlasTextureBuffer[i];
                colors[i] = new Color32(c, c, c, c);
            }

            // Clear allocation of 
            m_AtlasTextureBuffer = null;

            if ((m_GlyphRenderMode & GlyphRenderMode.RASTER) == GlyphRenderMode.RASTER || (m_GlyphRenderMode & GlyphRenderMode.RASTER_HINTED) == GlyphRenderMode.RASTER_HINTED)
                m_FontAtlasTexture.filterMode = FilterMode.Point;

            m_FontAtlasTexture.SetPixels32(colors, 0);
            m_FontAtlasTexture.Apply(false, false);

            // Saving File for Debug
            //var pngData = m_FontAtlasTexture.EncodeToPNG();
            //File.WriteAllBytes("Assets/Textures/Debug Font Texture.png", pngData);
        }


        /// <summary>
        /// Open Save Dialog to provide the option save the font asset using the name of the source font file. This also appends SDF to the name if using any of the SDF Font Asset creation modes.
        /// </summary>
        /// <param name="sourceObject"></param>
        void SaveNewFontAsset(Object sourceObject)
        {
            string filePath;
            
            // Save new Font Asset and open save file requester at Source Font File location.
            string saveDirectory = new FileInfo(AssetDatabase.GetAssetPath(sourceObject)).DirectoryName;

            if (((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP)
            {
                filePath = EditorUtility.SaveFilePanel("Save TextMesh Pro! Font Asset File", saveDirectory, sourceObject.name, "asset");

                if (filePath.Length == 0)
                    return;

                Save_Bitmap_FontAsset(filePath);
            }
            else
            {
                filePath = EditorUtility.SaveFilePanel("Save TextMesh Pro! Font Asset File", saveDirectory, sourceObject.name + " SDF", "asset");

                if (filePath.Length == 0)
                    return;

                Save_SDF_FontAsset(filePath);
            }
        }


        /// <summary>
        /// Open Save Dialog to provide the option to save the font asset under the same name.
        /// </summary>
        /// <param name="sourceObject"></param>
        void SaveNewFontAssetWithSameName(Object sourceObject)
        {
            string filePath;

            // Save new Font Asset and open save file requester at Source Font File location.
            string saveDirectory = new FileInfo(AssetDatabase.GetAssetPath(sourceObject)).DirectoryName;

            filePath = EditorUtility.SaveFilePanel("Save TextMesh Pro! Font Asset File", saveDirectory, sourceObject.name, "asset");

            if (filePath.Length == 0)
                return;

            if (((GlyphRasterModes)m_GlyphRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP)
            {
                Save_Bitmap_FontAsset(filePath);
            }
            else
            {
                Save_SDF_FontAsset(filePath);
            }
        }


        void Save_Bitmap_FontAsset(string filePath)
        {
            filePath = filePath.Substring(0, filePath.Length - 6); // Trim file extension from filePath.

            string dataPath = Application.dataPath;

            if (filePath.IndexOf(dataPath, System.StringComparison.InvariantCultureIgnoreCase) == -1)
            {
                Debug.LogError("You're saving the font asset in a directory outside of this project folder. This is not supported. Please select a directory under \"" + dataPath + "\"");
                return;
            }

            string relativeAssetPath = filePath.Substring(dataPath.Length - 6);
            string tex_DirName = Path.GetDirectoryName(relativeAssetPath);
            string tex_FileName = Path.GetFileNameWithoutExtension(relativeAssetPath);
            string tex_Path_NoExt = tex_DirName + "/" + tex_FileName;

            // Check if TextMeshPro font asset already exists. If not, create a new one. Otherwise update the existing one.
            TMP_FontAsset fontAsset = AssetDatabase.LoadAssetAtPath(tex_Path_NoExt + ".asset", typeof(TMP_FontAsset)) as TMP_FontAsset;
            if (fontAsset == null)
            {
                //Debug.Log("Creating TextMeshPro font asset!");
                fontAsset = ScriptableObject.CreateInstance<TMP_FontAsset>(); // Create new TextMeshPro Font Asset.
                AssetDatabase.CreateAsset(fontAsset, tex_Path_NoExt + ".asset");

                // Set version number of font asset
                fontAsset.version = "1.1.0";

                //Set Font Asset Type
                fontAsset.atlasRenderMode = m_GlyphRenderMode;

                // Reference to the source font file GUID.
                fontAsset.m_SourceFontFileGUID = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(m_SourceFontFile));

                // Add FaceInfo to Font Asset
                fontAsset.faceInfo = m_FaceInfo;

                // Add GlyphInfo[] to Font Asset
                fontAsset.glyphTable = m_FontGlyphTable;

                // Add CharacterTable[] to font asset.
                fontAsset.characterTable = m_FontCharacterTable;

                // Sort glyph and character tables.
                fontAsset.SortGlyphAndCharacterTables();

                // Get and Add Kerning Pairs to Font Asset
                if (m_IncludeFontFeatures)
                    fontAsset.fontFeatureTable = GetKerningTable();


                // Add Font Atlas as Sub-Asset
                fontAsset.atlasTextures = new Texture2D[] { m_FontAtlasTexture };
                m_FontAtlasTexture.name = tex_FileName + " Atlas";
                fontAsset.atlasWidth = m_AtlasWidth;
                fontAsset.atlasHeight = m_AtlasHeight;
                fontAsset.atlasPadding = m_Padding;

                AssetDatabase.AddObjectToAsset(m_FontAtlasTexture, fontAsset);

                // Create new Material and Add it as Sub-Asset
                Shader default_Shader = Shader.Find("TextMeshPro/Bitmap"); // m_shaderSelection;
                Material tmp_material = new Material(default_Shader);
                tmp_material.name = tex_FileName + " Material";
                tmp_material.SetTexture(ShaderUtilities.ID_MainTex, m_FontAtlasTexture);
                fontAsset.material = tmp_material;

                AssetDatabase.AddObjectToAsset(tmp_material, fontAsset);

            }
            else
            {
                // Find all Materials referencing this font atlas.
                Material[] material_references = TMP_EditorUtility.FindMaterialReferences(fontAsset);

                // Set version number of font asset
                fontAsset.version = "1.1.0";

                // Special handling to remove legacy font asset data
                if (fontAsset.m_glyphInfoList != null && fontAsset.m_glyphInfoList.Count > 0)
                    fontAsset.m_glyphInfoList = null;

                // Destroy Assets that will be replaced.
                if (fontAsset.atlasTextures != null && fontAsset.atlasTextures.Length > 0)
                    DestroyImmediate(fontAsset.atlasTextures[0], true);

                //Set Font Asset Type
                fontAsset.atlasRenderMode = m_GlyphRenderMode;

                // Add FaceInfo to Font Asset
                fontAsset.faceInfo = m_FaceInfo;

                // Add GlyphInfo[] to Font Asset
                fontAsset.glyphTable = m_FontGlyphTable;

                // Add CharacterTable[] to font asset.
                fontAsset.characterTable = m_FontCharacterTable;

                // Sort glyph and character tables.
                fontAsset.SortGlyphAndCharacterTables();

                // Get and Add Kerning Pairs to Font Asset
                if (m_IncludeFontFeatures)
                    fontAsset.fontFeatureTable = GetKerningTable();

                // Add Font Atlas as Sub-Asset
                fontAsset.atlasTextures = new Texture2D[] { m_FontAtlasTexture };
                m_FontAtlasTexture.name = tex_FileName + " Atlas";
                fontAsset.atlasWidth = m_AtlasWidth;
                fontAsset.atlasHeight = m_AtlasHeight;
                fontAsset.atlasPadding = m_Padding;

                // Special handling due to a bug in earlier versions of Unity.
                m_FontAtlasTexture.hideFlags = HideFlags.None;
                fontAsset.material.hideFlags = HideFlags.None;

                AssetDatabase.AddObjectToAsset(m_FontAtlasTexture, fontAsset);

                // Assign new font atlas texture to the existing material.
                fontAsset.material.SetTexture(ShaderUtilities.ID_MainTex, fontAsset.atlasTextures[0]);

                // Update the Texture reference on the Material
                for (int i = 0; i < material_references.Length; i++)
                {
                    material_references[i].SetTexture(ShaderUtilities.ID_MainTex, m_FontAtlasTexture);
                }
            }

            // Add list of GlyphRects to font asset.
            fontAsset.freeGlyphRects = m_FreeGlyphRects;
            fontAsset.usedGlyphRects = m_UsedGlyphRects;

            // Save Font Asset creation settings
            m_SelectedFontAsset = fontAsset;
            m_LegacyFontAsset = null;
            fontAsset.creationSettings = SaveFontCreationSettings();

            AssetDatabase.SaveAssets();

            AssetDatabase.ImportAsset(AssetDatabase.GetAssetPath(fontAsset));  // Re-import font asset to get the new updated version.

            //EditorUtility.SetDirty(font_asset);
            fontAsset.ReadFontAssetDefinition();

            AssetDatabase.Refresh();

            m_FontAtlasTexture = null;

            // NEED TO GENERATE AN EVENT TO FORCE A REDRAW OF ANY TEXTMESHPRO INSTANCES THAT MIGHT BE USING THIS FONT ASSET
            TMPro_EventManager.ON_FONT_PROPERTY_CHANGED(true, fontAsset);
        }


        void Save_SDF_FontAsset(string filePath)
        {
            filePath = filePath.Substring(0, filePath.Length - 6); // Trim file extension from filePath.

            string dataPath = Application.dataPath;

            if (filePath.IndexOf(dataPath, System.StringComparison.InvariantCultureIgnoreCase) == -1)
            {
                Debug.LogError("You're saving the font asset in a directory outside of this project folder. This is not supported. Please select a directory under \"" + dataPath + "\"");
                return;
            }

            string relativeAssetPath = filePath.Substring(dataPath.Length - 6);
            string tex_DirName = Path.GetDirectoryName(relativeAssetPath);
            string tex_FileName = Path.GetFileNameWithoutExtension(relativeAssetPath);
            string tex_Path_NoExt = tex_DirName + "/" + tex_FileName;


            // Check if TextMeshPro font asset already exists. If not, create a new one. Otherwise update the existing one.
            TMP_FontAsset fontAsset = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(tex_Path_NoExt + ".asset");
            if (fontAsset == null)
            {
                //Debug.Log("Creating TextMeshPro font asset!");
                fontAsset = ScriptableObject.CreateInstance<TMP_FontAsset>(); // Create new TextMeshPro Font Asset.
                AssetDatabase.CreateAsset(fontAsset, tex_Path_NoExt + ".asset");

                // Set version number of font asset
                fontAsset.version = "1.1.0";

                // Reference to source font file GUID.
                fontAsset.m_SourceFontFileGUID = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(m_SourceFontFile));

                //Set Font Asset Type
                fontAsset.atlasRenderMode = m_GlyphRenderMode;

                // Add FaceInfo to Font Asset
                fontAsset.faceInfo = m_FaceInfo;

                // Add GlyphInfo[] to Font Asset
                fontAsset.glyphTable = m_FontGlyphTable;

                // Add CharacterTable[] to font asset.
                fontAsset.characterTable = m_FontCharacterTable;

                // Sort glyph and character tables.
                fontAsset.SortGlyphAndCharacterTables();

                // Get and Add Kerning Pairs to Font Asset
                if (m_IncludeFontFeatures)
                    fontAsset.fontFeatureTable = GetKerningTable();

                // Add Font Atlas as Sub-Asset
                fontAsset.atlasTextures = new Texture2D[] { m_FontAtlasTexture };
                m_FontAtlasTexture.name = tex_FileName + " Atlas";
                fontAsset.atlasWidth = m_AtlasWidth;
                fontAsset.atlasHeight = m_AtlasHeight;
                fontAsset.atlasPadding = m_Padding;

                AssetDatabase.AddObjectToAsset(m_FontAtlasTexture, fontAsset);

                // Create new Material and Add it as Sub-Asset
                Shader default_Shader = Shader.Find("TextMeshPro/Distance Field");
                Material tmp_material = new Material(default_Shader);

                tmp_material.name = tex_FileName + " Material";
                tmp_material.SetTexture(ShaderUtilities.ID_MainTex, m_FontAtlasTexture);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureWidth, m_FontAtlasTexture.width);
                tmp_material.SetFloat(ShaderUtilities.ID_TextureHeight, m_FontAtlasTexture.height);

                int spread = m_Padding + 1;
                tmp_material.SetFloat(ShaderUtilities.ID_GradientScale, spread); // Spread = Padding for Brute Force SDF.

                tmp_material.SetFloat(ShaderUtilities.ID_WeightNormal, fontAsset.normalStyle);
                tmp_material.SetFloat(ShaderUtilities.ID_WeightBold, fontAsset.boldStyle);

                fontAsset.material = tmp_material;

                AssetDatabase.AddObjectToAsset(tmp_material, fontAsset);

            }
            else
            {
                // Find all Materials referencing this font atlas.
                Material[] material_references = TMP_EditorUtility.FindMaterialReferences(fontAsset);

                // Destroy Assets that will be replaced.
                if (fontAsset.atlasTextures != null && fontAsset.atlasTextures.Length > 0)
                    DestroyImmediate(fontAsset.atlasTextures[0], true);

                // Set version number of font asset
                fontAsset.version = "1.1.0";

                // Special handling to remove legacy font asset data
                if (fontAsset.m_glyphInfoList != null && fontAsset.m_glyphInfoList.Count > 0)
                    fontAsset.m_glyphInfoList = null;

                //Set Font Asset Type
                fontAsset.atlasRenderMode = m_GlyphRenderMode;

                // Add FaceInfo to Font Asset  
                fontAsset.faceInfo = m_FaceInfo;

                // Add GlyphInfo[] to Font Asset
                fontAsset.glyphTable = m_FontGlyphTable;

                // Add CharacterTable[] to font asset.
                fontAsset.characterTable = m_FontCharacterTable;

                // Sort glyph and character tables.
                fontAsset.SortGlyphAndCharacterTables();

                // Get and Add Kerning Pairs to Font Asset
                // TODO: Check and preserve existing adjustment pairs.
                if (m_IncludeFontFeatures)
                    fontAsset.fontFeatureTable = GetKerningTable();

                // Add Font Atlas as Sub-Asset
                fontAsset.atlasTextures = new Texture2D[] { m_FontAtlasTexture };
                m_FontAtlasTexture.name = tex_FileName + " Atlas";
                fontAsset.atlasWidth = m_AtlasWidth;
                fontAsset.atlasHeight = m_AtlasHeight;
                fontAsset.atlasPadding = m_Padding;

                // Special handling due to a bug in earlier versions of Unity.
                m_FontAtlasTexture.hideFlags = HideFlags.None;
                fontAsset.material.hideFlags = HideFlags.None;

                AssetDatabase.AddObjectToAsset(m_FontAtlasTexture, fontAsset);

                // Assign new font atlas texture to the existing material.
                fontAsset.material.SetTexture(ShaderUtilities.ID_MainTex, fontAsset.atlasTextures[0]);

                // Update the Texture reference on the Material
                for (int i = 0; i < material_references.Length; i++)
                {
                    material_references[i].SetTexture(ShaderUtilities.ID_MainTex, m_FontAtlasTexture);
                    material_references[i].SetFloat(ShaderUtilities.ID_TextureWidth, m_FontAtlasTexture.width);
                    material_references[i].SetFloat(ShaderUtilities.ID_TextureHeight, m_FontAtlasTexture.height);

                    int spread = m_Padding + 1;
                    material_references[i].SetFloat(ShaderUtilities.ID_GradientScale, spread); // Spread = Padding for Brute Force SDF.

                    material_references[i].SetFloat(ShaderUtilities.ID_WeightNormal, fontAsset.normalStyle);
                    material_references[i].SetFloat(ShaderUtilities.ID_WeightBold, fontAsset.boldStyle);
                }
            }

            // Saving File for Debug
            //var pngData = destination_Atlas.EncodeToPNG();
            //File.WriteAllBytes("Assets/Textures/Debug Distance Field.png", pngData);

            // Add list of GlyphRects to font asset.
            fontAsset.freeGlyphRects = m_FreeGlyphRects;
            fontAsset.usedGlyphRects = m_UsedGlyphRects;

            // Save Font Asset creation settings
            m_SelectedFontAsset = fontAsset;
            m_LegacyFontAsset = null;
            fontAsset.creationSettings = SaveFontCreationSettings();

            AssetDatabase.SaveAssets();

            AssetDatabase.ImportAsset(AssetDatabase.GetAssetPath(fontAsset));  // Re-import font asset to get the new updated version.

            fontAsset.ReadFontAssetDefinition();

            AssetDatabase.Refresh();

            m_FontAtlasTexture = null;

            // NEED TO GENERATE AN EVENT TO FORCE A REDRAW OF ANY TEXTMESHPRO INSTANCES THAT MIGHT BE USING THIS FONT ASSET
            TMPro_EventManager.ON_FONT_PROPERTY_CHANGED(true, fontAsset);
        }


        /// <summary>
        /// Internal method to save the Font Asset Creation Settings
        /// </summary>
        /// <returns></returns>
        FontAssetCreationSettings SaveFontCreationSettings()
        {
            FontAssetCreationSettings settings = new FontAssetCreationSettings();

            //settings.sourceFontFileName = m_SourceFontFile.name;
            settings.sourceFontFileGUID = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(m_SourceFontFile));
            settings.pointSizeSamplingMode = m_PointSizeSamplingMode;
            settings.pointSize = m_PointSize;
            settings.padding = m_Padding;
            settings.packingMode = (int)m_PackingMode;
            settings.atlasWidth = m_AtlasWidth;
            settings.atlasHeight = m_AtlasHeight;
            settings.characterSetSelectionMode = m_CharacterSetSelectionMode;
            settings.characterSequence = m_CharacterSequence;
            settings.referencedFontAssetGUID = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(m_ReferencedFontAsset));
            settings.referencedTextAssetGUID = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(m_CharactersFromFile));
            //settings.fontStyle = (int)m_FontStyle;
            //settings.fontStyleModifier = m_FontStyleValue;
            settings.renderMode = (int)m_GlyphRenderMode;
            settings.includeFontFeatures = m_IncludeFontFeatures;

            return settings;
        }


        /// <summary>
        /// Internal method to load the Font Asset Creation Settings
        /// </summary>
        /// <param name="settings"></param>
        void LoadFontCreationSettings(FontAssetCreationSettings settings)
        {
            m_SourceFontFile = AssetDatabase.LoadAssetAtPath<Font>(AssetDatabase.GUIDToAssetPath(settings.sourceFontFileGUID));
            m_PointSizeSamplingMode  = settings.pointSizeSamplingMode;
            m_PointSize = settings.pointSize;
            m_Padding = settings.padding;
            m_PackingMode = (FontPackingModes)settings.packingMode;
            m_AtlasWidth = settings.atlasWidth;
            m_AtlasHeight = settings.atlasHeight;
            m_CharacterSetSelectionMode = settings.characterSetSelectionMode;
            m_CharacterSequence = settings.characterSequence;
            m_ReferencedFontAsset = AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(AssetDatabase.GUIDToAssetPath(settings.referencedFontAssetGUID));
            m_CharactersFromFile = AssetDatabase.LoadAssetAtPath<TextAsset>(AssetDatabase.GUIDToAssetPath(settings.referencedTextAssetGUID));
            //m_FontStyle = (FaceStyles)settings.fontStyle;
            //m_FontStyleValue = settings.fontStyleModifier;
            m_GlyphRenderMode = (GlyphRenderMode)settings.renderMode;
            m_IncludeFontFeatures = settings.includeFontFeatures;
        }


        /// <summary>
        /// Save the latest font asset creation settings to EditorPrefs.
        /// </summary>
        /// <param name="settings"></param>
        void SaveCreationSettingsToEditorPrefs(FontAssetCreationSettings settings)
        {
            // Create new list if one does not already exist
            if (m_FontAssetCreationSettingsContainer == null)
            {
                m_FontAssetCreationSettingsContainer = new FontAssetCreationSettingsContainer();
                m_FontAssetCreationSettingsContainer.fontAssetCreationSettings = new List<FontAssetCreationSettings>();
            }

            // Add new creation settings to the list
            m_FontAssetCreationSettingsContainer.fontAssetCreationSettings.Add(settings);

            // Since list should only contain the most 4 recent settings, we remove the first element if list exceeds 4 elements.
            if (m_FontAssetCreationSettingsContainer.fontAssetCreationSettings.Count > 4)
                m_FontAssetCreationSettingsContainer.fontAssetCreationSettings.RemoveAt(0);

            m_FontAssetCreationSettingsCurrentIndex = m_FontAssetCreationSettingsContainer.fontAssetCreationSettings.Count - 1;

            // Serialize list to JSON
            string serializedSettings = JsonUtility.ToJson(m_FontAssetCreationSettingsContainer, true);

            EditorPrefs.SetString(k_FontAssetCreationSettingsContainerKey, serializedSettings);
        }

        void DrawPreview()
        {
            Rect pixelRect;
            if (position.width > position.height && position.width > k_TwoColumnControlsWidth)
            {
                float minSide = Mathf.Min(position.height - 15f, position.width - k_TwoColumnControlsWidth);

                EditorGUILayout.BeginVertical(EditorStyles.helpBox, GUILayout.MaxWidth(minSide));

                pixelRect = GUILayoutUtility.GetRect(minSide, minSide, GUILayout.ExpandHeight(false), GUILayout.ExpandWidth(false));
            }
            else
            {
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);

                pixelRect = GUILayoutUtility.GetAspectRect(1f);
            }
            
            if (m_FontAtlasTexture != null)
            {
                EditorGUI.DrawTextureAlpha(pixelRect, m_FontAtlasTexture, ScaleMode.StretchToFill);
            }
            else if (m_SavedFontAtlas != null)
            {
                EditorGUI.DrawTextureAlpha(pixelRect, m_SavedFontAtlas, ScaleMode.StretchToFill);
            }

            EditorGUILayout.EndVertical();
        }


        void CheckForLegacyGlyphRenderMode()
        {
            // Special handling for legacy glyph render mode
            if ((int)m_GlyphRenderMode < 0x100)
            {
                switch ((int)m_GlyphRenderMode)
                {
                    case 0:
                        m_GlyphRenderMode = GlyphRenderMode.SMOOTH_HINTED;
                        break;
                    case 1:
                        m_GlyphRenderMode = GlyphRenderMode.SMOOTH;
                        break;
                    case 2:
                        m_GlyphRenderMode = GlyphRenderMode.RASTER_HINTED;
                        break;
                    case 3:
                        m_GlyphRenderMode = GlyphRenderMode.RASTER;
                        break;
                    case 6:
                    case 7:
                        m_GlyphRenderMode = GlyphRenderMode.SDFAA;
                        break;
                }
            }
        }


        // Get Kerning Pairs
        public TMP_FontFeatureTable GetKerningTable()
        {
            GlyphPairAdjustmentRecord[] adjustmentRecords = FontEngine.GetGlyphPairAdjustmentTable(m_AvailableGlyphsToAdd.ToArray());

            if (adjustmentRecords == null)
                return null;

            TMP_FontFeatureTable fontFeatureTable = new TMP_FontFeatureTable();

            for (int i = 0; i < adjustmentRecords.Length; i++)
            {
                fontFeatureTable.glyphPairAdjustmentRecords.Add(new TMP_GlyphPairAdjustmentRecord(adjustmentRecords[i]));
            }

            fontFeatureTable.SortGlyphPairAdjustmentRecords();

            return fontFeatureTable;
        }
    }
}