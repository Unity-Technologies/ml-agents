using UnityEngine;
using System.Collections;
using System.Collections.Generic;


#pragma warning disable 0649 // Disabled warnings related to serialized fields not assigned in this script but used in the editor.

namespace TMPro
{
    [System.Serializable]
    public class TMP_Settings : ScriptableObject
    {
        private static TMP_Settings s_Instance;

        /// <summary>
        /// Returns the release version of the product.
        /// </summary>
        public static string version
        {
            get { return "1.4.0"; }
        }

        /// <summary>
        /// Controls if Word Wrapping will be enabled on newly created text objects by default.
        /// </summary>
        public static bool enableWordWrapping
        {
            get { return instance.m_enableWordWrapping; }
        }
        [SerializeField]
        private bool m_enableWordWrapping;

        /// <summary>
        /// Controls if Kerning is enabled on newly created text objects by default.
        /// </summary>
        public static bool enableKerning
        {
            get { return instance.m_enableKerning; }
        }
        [SerializeField]
        private bool m_enableKerning;

        /// <summary>
        /// Controls if Extra Padding is enabled on newly created text objects by default.
        /// </summary>
        public static bool enableExtraPadding
        {
            get { return instance.m_enableExtraPadding; }
        }
        [SerializeField]
        private bool m_enableExtraPadding;

        /// <summary>
        /// Controls if TintAllSprites is enabled on newly created text objects by default.
        /// </summary>
        public static bool enableTintAllSprites
        {
            get { return instance.m_enableTintAllSprites; }
        }
        [SerializeField]
        private bool m_enableTintAllSprites;

        /// <summary>
        /// Controls if Escape Characters will be parsed in the Text Input Box on newly created text objects.
        /// </summary>
        public static bool enableParseEscapeCharacters
        {
            get { return instance.m_enableParseEscapeCharacters; }
        }
        [SerializeField]
        private bool m_enableParseEscapeCharacters;

        /// <summary>
        /// Controls if Raycast Target is enabled by default on newly created text objects.
        /// </summary>
        public static bool enableRaycastTarget
        {
            get { return instance.m_EnableRaycastTarget; }
        }
        [SerializeField]
        private bool m_EnableRaycastTarget = true;

        /// <summary>
        /// Determines if OpenType Font Features should be retrieved at runtime from the source font file.
        /// </summary>
        public static bool getFontFeaturesAtRuntime
        {
            get { return instance.m_GetFontFeaturesAtRuntime; }
        }
        [SerializeField]
        private bool m_GetFontFeaturesAtRuntime = true;

        /// <summary>
        /// The character that will be used as a replacement for missing glyphs in a font asset.
        /// </summary>
        public static int missingGlyphCharacter
        {
            get { return instance.m_missingGlyphCharacter; }
            set { instance.m_missingGlyphCharacter = value; }
        }
        [SerializeField]
        private int m_missingGlyphCharacter;

        /// <summary>
        /// Controls the display of warning message in the console.
        /// </summary>
        public static bool warningsDisabled
        {
            get { return instance.m_warningsDisabled; }
        }
        [SerializeField]
        private bool m_warningsDisabled;

        /// <summary>
        /// Returns the Default Font Asset to be used by newly created text objects.
        /// </summary>
        public static TMP_FontAsset defaultFontAsset
        {
            get { return instance.m_defaultFontAsset; }
        }
        [SerializeField]
        private TMP_FontAsset m_defaultFontAsset;

        /// <summary>
        /// The relative path to a Resources folder in the project.
        /// </summary>
        public static string defaultFontAssetPath
        {
            get { return instance.m_defaultFontAssetPath; }
        }
        [SerializeField]
        private string m_defaultFontAssetPath;

        /// <summary>
        /// The Default Point Size of newly created text objects.
        /// </summary>
        public static float defaultFontSize
        {
            get { return instance.m_defaultFontSize; }
        }
        [SerializeField]
        private float m_defaultFontSize;

        /// <summary>
        /// The multiplier used to computer the default Min point size when Text Auto Sizing is used.
        /// </summary>
        public static float defaultTextAutoSizingMinRatio
        {
            get { return instance.m_defaultAutoSizeMinRatio; }
        }
        [SerializeField]
        private float m_defaultAutoSizeMinRatio;

        /// <summary>
        /// The multiplier used to computer the default Max point size when Text Auto Sizing is used.
        /// </summary>
        public static float defaultTextAutoSizingMaxRatio
        {
            get { return instance.m_defaultAutoSizeMaxRatio; }
        }
        [SerializeField]
        private float m_defaultAutoSizeMaxRatio;

        /// <summary>
        /// The Default Size of the Text Container of a TextMeshPro object.
        /// </summary>
        public static Vector2 defaultTextMeshProTextContainerSize
        {
            get { return instance.m_defaultTextMeshProTextContainerSize; }
        }
        [SerializeField]
        private Vector2 m_defaultTextMeshProTextContainerSize;

        /// <summary>
        /// The Default Width of the Text Container of a TextMeshProUI object.
        /// </summary>
        public static Vector2 defaultTextMeshProUITextContainerSize
        {
            get { return instance.m_defaultTextMeshProUITextContainerSize; }
        }
        [SerializeField]
        private Vector2 m_defaultTextMeshProUITextContainerSize;

        /// <summary>
        /// Set the size of the text container of newly created text objects to match the size of the text.
        /// </summary>
        public static bool autoSizeTextContainer
        {
            get { return instance.m_autoSizeTextContainer; }
        }
        [SerializeField]
        private bool m_autoSizeTextContainer;

        /// <summary>
        /// Returns the list of Fallback Fonts defined in the TMP Settings file.
        /// </summary>
        public static List<TMP_FontAsset> fallbackFontAssets
        {
            get { return instance.m_fallbackFontAssets; }
        }
        [SerializeField]
        private List<TMP_FontAsset> m_fallbackFontAssets;

        /// <summary>
        /// Controls whether or not TMP will create a matching material preset or use the default material of the fallback font asset.
        /// </summary>
        public static bool matchMaterialPreset
        {
            get { return instance.m_matchMaterialPreset; }
        }
        [SerializeField]
        private bool m_matchMaterialPreset;

        /// <summary>
        /// The Default Sprite Asset to be used by default.
        /// </summary>
        public static TMP_SpriteAsset defaultSpriteAsset
        {
            get { return instance.m_defaultSpriteAsset; }
        }
        [SerializeField]
        private TMP_SpriteAsset m_defaultSpriteAsset;

        /// <summary>
        /// The relative path to a Resources folder in the project.
        /// </summary>
        public static string defaultSpriteAssetPath
        {
            get { return instance.m_defaultSpriteAssetPath; }
        }
        [SerializeField]
        private string m_defaultSpriteAssetPath;

        /// <summary>
        /// The relative path to a Resources folder in the project that contains Color Gradient Presets.
        /// </summary>
        public static string defaultColorGradientPresetsPath
        {
            get { return instance.m_defaultColorGradientPresetsPath; }
        }
        [SerializeField]
        private string m_defaultColorGradientPresetsPath;

        /// <summary>
        /// Determines if Emoji support is enabled in the Input Field TouchScreenKeyboard.
        /// </summary>
        public static bool enableEmojiSupport
        {
            get { return instance.m_enableEmojiSupport; }
            set { instance.m_enableEmojiSupport = value; }
        }
        [SerializeField]
        private bool m_enableEmojiSupport;

        /// <summary>
        /// The Default Style Sheet used by the text objects.
        /// </summary>
        public static TMP_StyleSheet defaultStyleSheet
        {
            get { return instance.m_defaultStyleSheet; }
        }
        [SerializeField]
        private TMP_StyleSheet m_defaultStyleSheet;

        /// <summary>
        /// Text file that contains the leading characters used for line breaking for Asian languages.
        /// </summary>
        public static TextAsset leadingCharacters
        {
            get { return instance.m_leadingCharacters; }
        }
        [SerializeField]
        private TextAsset m_leadingCharacters;

        /// <summary>
        /// Text file that contains the following characters used for line breaking for Asian languages.
        /// </summary>
        public static TextAsset followingCharacters
        {
            get { return instance.m_followingCharacters; }
        }
        [SerializeField]
        private TextAsset m_followingCharacters;

        /// <summary>
        /// 
        /// </summary>
        public static LineBreakingTable linebreakingRules
        {
            get
            {
                if (instance.m_linebreakingRules == null)
                    LoadLinebreakingRules();

                return instance.m_linebreakingRules;
            }
        }
        [SerializeField]
        private LineBreakingTable m_linebreakingRules;

        /// <summary>
        /// Get a singleton instance of the settings class.
        /// </summary>
        public static TMP_Settings instance
        {
            get
            {
                if (TMP_Settings.s_Instance == null)
                {
                    TMP_Settings.s_Instance = Resources.Load<TMP_Settings>("TMP Settings");

                    #if UNITY_EDITOR
                    // Make sure TextMesh Pro UPM packages resources have been added to the user project
                    if (TMP_Settings.s_Instance == null)
                    {
                        // Open TMP Resources Importer
                        TMP_PackageResourceImporterWindow.ShowPackageImporterWindow();
                    }
                    #endif
                }

                return TMP_Settings.s_Instance;
            }
        }


        /// <summary>
        /// Static Function to load the TMP Settings file.
        /// </summary>
        /// <returns></returns>
        public static TMP_Settings LoadDefaultSettings()
        {
            if (s_Instance == null)
            {
                // Load settings from TMP_Settings file
                TMP_Settings settings = Resources.Load<TMP_Settings>("TMP Settings");
                if (settings != null)
                    s_Instance = settings;
            }

            return s_Instance;
        }


        /// <summary>
        /// Returns the Sprite Asset defined in the TMP Settings file.
        /// </summary>
        /// <returns></returns>
        public static TMP_Settings GetSettings()
        {
            if (TMP_Settings.instance == null) return null;

            return TMP_Settings.instance;
        }


        /// <summary>
        /// Returns the Font Asset defined in the TMP Settings file.
        /// </summary>
        /// <returns></returns>
        public static TMP_FontAsset GetFontAsset()
        {
            if (TMP_Settings.instance == null) return null;

            return TMP_Settings.instance.m_defaultFontAsset;
        }


        /// <summary>
        /// Returns the Sprite Asset defined in the TMP Settings file.
        /// </summary>
        /// <returns></returns>
        public static TMP_SpriteAsset GetSpriteAsset()
        {
            if (TMP_Settings.instance == null) return null;

            return TMP_Settings.instance.m_defaultSpriteAsset;
        }


        /// <summary>
        /// Returns the Sprite Asset defined in the TMP Settings file.
        /// </summary>
        /// <returns></returns>
        public static TMP_StyleSheet GetStyleSheet()
        {
            if (TMP_Settings.instance == null) return null;

            return TMP_Settings.instance.m_defaultStyleSheet;
        }


        public static void LoadLinebreakingRules()
        {
            //Debug.Log("Loading Line Breaking Rules for Asian Languages.");

            if (TMP_Settings.instance == null) return;

            if (s_Instance.m_linebreakingRules == null)
                s_Instance.m_linebreakingRules = new LineBreakingTable();

            s_Instance.m_linebreakingRules.leadingCharacters = GetCharacters(s_Instance.m_leadingCharacters);
            s_Instance.m_linebreakingRules.followingCharacters = GetCharacters(s_Instance.m_followingCharacters);
        }


        /// <summary>
        ///  Get the characters from the line breaking files
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        private static Dictionary<int, char> GetCharacters(TextAsset file)
        {
            Dictionary<int, char> dict = new Dictionary<int, char>();
            string text = file.text;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                // Check to make sure we don't include duplicates
                if (dict.ContainsKey((int)c) == false)
                {
                    dict.Add((int)c, c);
                    //Debug.Log("Adding [" + (int)c + "] to dictionary.");
                }
                //else
                //    Debug.Log("Character [" + text[i] + "] is a duplicate.");
            }

            return dict;
        }


        public class LineBreakingTable
        {
            public Dictionary<int, char> leadingCharacters;
            public Dictionary<int, char> followingCharacters;
        }
    }
}
