using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

#pragma warning disable 0414 // Disabled a few warnings for not yet implemented features.

namespace TMPro.EditorUtilities
{
    [CustomEditor(typeof(TMP_Settings))]
    public class TMP_SettingsEditor : Editor
    {
        internal class Styles
        {
            public static readonly GUIContent defaultFontAssetLabel = new GUIContent("Default Font Asset", "The Font Asset that will be assigned by default to newly created text objects when no Font Asset is specified.");
            public static readonly GUIContent defaultFontAssetPathLabel = new GUIContent("Path:        Resources/", "The relative path to a Resources folder where the Font Assets and Material Presets are located.\nExample \"Fonts & Materials/\"");

            public static readonly GUIContent fallbackFontAssetsLabel = new GUIContent("Fallback Font Assets", "The Font Assets that will be searched to locate and replace missing characters from a given Font Asset.");
            public static readonly GUIContent fallbackFontAssetsListLabel = new GUIContent("Fallback Font Assets List", "The Font Assets that will be searched to locate and replace missing characters from a given Font Asset.");

            public static readonly GUIContent fallbackMaterialSettingsLabel = new GUIContent("Fallback Material Settings");
            public static readonly GUIContent matchMaterialPresetLabel = new GUIContent("Match Material Presets");

            public static readonly GUIContent containerDefaultSettingsLabel = new GUIContent("Text Container Default Settings");

            public static readonly GUIContent textMeshProLabel = new GUIContent("TextMeshPro");
            public static readonly GUIContent textMeshProUiLabel = new GUIContent("TextMeshPro UI");
            public static readonly GUIContent enableRaycastTarget = new GUIContent("Enable Raycast Target");
            public static readonly GUIContent autoSizeContainerLabel = new GUIContent("Auto Size Text Container", "Set the size of the text container to match the text.");

            public static readonly GUIContent textComponentDefaultSettingsLabel = new GUIContent("Text Component Default Settings");
            public static readonly GUIContent defaultFontSize = new GUIContent("Default Font Size");
            public static readonly GUIContent autoSizeRatioLabel = new GUIContent("Text Auto Size Ratios");
            public static readonly GUIContent minLabel = new GUIContent("Min");
            public static readonly GUIContent maxLabel = new GUIContent("Max");

            public static readonly GUIContent wordWrappingLabel = new GUIContent("Word Wrapping");
            public static readonly GUIContent kerningLabel = new GUIContent("Kerning");
            public static readonly GUIContent extraPaddingLabel = new GUIContent("Extra Padding");
            public static readonly GUIContent tintAllSpritesLabel = new GUIContent("Tint All Sprites");
            public static readonly GUIContent parseEscapeCharactersLabel = new GUIContent("Parse Escape Sequence");

            public static readonly GUIContent dynamicFontSystemSettingsLabel = new GUIContent("Dynamic Font System Settings");
            public static readonly GUIContent getFontFeaturesAtRuntime = new GUIContent("Get Font Features at Runtime", "Determines if Glyph Adjustment Data will be retrieved from font files at runtime when new characters and glyphs are added to font assets.");

            public static readonly GUIContent missingGlyphLabel = new GUIContent("Replacement Character", "The character to be displayed when the requested character is not found in any font asset or fallbacks.");
            public static readonly GUIContent disableWarningsLabel = new GUIContent("Disable warnings", "Disable warning messages in the Console.");

            public static readonly GUIContent defaultSpriteAssetLabel = new GUIContent("Default Sprite Asset", "The Sprite Asset that will be assigned by default when using the <sprite> tag when no Sprite Asset is specified.");
            public static readonly GUIContent enableEmojiSupportLabel = new GUIContent("iOS Emoji Support", "Enables Emoji support for Touch Screen Keyboards on target devices.");
            public static readonly GUIContent spriteAssetsPathLabel = new GUIContent("Path:        Resources/", "The relative path to a Resources folder where the Sprite Assets are located.\nExample \"Sprite Assets/\"");

            public static readonly GUIContent defaultStyleSheetLabel = new GUIContent("Default Style Sheet", "The Style Sheet that will be used for all text objects in this project.");

            public static readonly GUIContent colorGradientPresetsLabel = new GUIContent("Color Gradient Presets", "The relative path to a Resources folder where the Color Gradient Presets are located.\nExample \"Color Gradient Presets/\"");
            public static readonly GUIContent colorGradientsPathLabel = new GUIContent("Path:        Resources/", "The relative path to a Resources folder where the Color Gradient Presets are located.\nExample \"Color Gradient Presets/\"");

            public static readonly GUIContent lineBreakingLabel = new GUIContent("Line Breaking for Asian languages", "The text assets that contain the Leading and Following characters which define the rules for line breaking with Asian languages.");
        }

        SerializedProperty m_PropFontAsset;
        SerializedProperty m_PropDefaultFontAssetPath;
        SerializedProperty m_PropDefaultFontSize;
        SerializedProperty m_PropDefaultAutoSizeMinRatio;
        SerializedProperty m_PropDefaultAutoSizeMaxRatio;
        SerializedProperty m_PropDefaultTextMeshProTextContainerSize;
        SerializedProperty m_PropDefaultTextMeshProUITextContainerSize;
        SerializedProperty m_PropAutoSizeTextContainer;
        SerializedProperty m_PropEnableRaycastTarget;

        SerializedProperty m_PropSpriteAsset;
        SerializedProperty m_PropSpriteAssetPath;
        SerializedProperty m_PropEnableEmojiSupport;
        SerializedProperty m_PropStyleSheet;
        ReorderableList m_List;

        SerializedProperty m_PropColorGradientPresetsPath;

        SerializedProperty m_PropMatchMaterialPreset;
        SerializedProperty m_PropWordWrapping;
        SerializedProperty m_PropKerning;
        SerializedProperty m_PropExtraPadding;
        SerializedProperty m_PropTintAllSprites;
        SerializedProperty m_PropParseEscapeCharacters;
        SerializedProperty m_PropMissingGlyphCharacter;

        SerializedProperty m_GetFontFeaturesAtRuntime;

        SerializedProperty m_PropWarningsDisabled;

        SerializedProperty m_PropLeadingCharacters;
        SerializedProperty m_PropFollowingCharacters;

        public void OnEnable()
        {
            if (target == null)
                return;

            m_PropFontAsset = serializedObject.FindProperty("m_defaultFontAsset");
            m_PropDefaultFontAssetPath = serializedObject.FindProperty("m_defaultFontAssetPath");
            m_PropDefaultFontSize = serializedObject.FindProperty("m_defaultFontSize");
            m_PropDefaultAutoSizeMinRatio = serializedObject.FindProperty("m_defaultAutoSizeMinRatio");
            m_PropDefaultAutoSizeMaxRatio = serializedObject.FindProperty("m_defaultAutoSizeMaxRatio");
            m_PropDefaultTextMeshProTextContainerSize = serializedObject.FindProperty("m_defaultTextMeshProTextContainerSize");
            m_PropDefaultTextMeshProUITextContainerSize = serializedObject.FindProperty("m_defaultTextMeshProUITextContainerSize");
            m_PropAutoSizeTextContainer = serializedObject.FindProperty("m_autoSizeTextContainer");
            m_PropEnableRaycastTarget = serializedObject.FindProperty("m_EnableRaycastTarget");

            m_PropSpriteAsset = serializedObject.FindProperty("m_defaultSpriteAsset");
            m_PropSpriteAssetPath = serializedObject.FindProperty("m_defaultSpriteAssetPath");
            m_PropEnableEmojiSupport = serializedObject.FindProperty("m_enableEmojiSupport");
            m_PropStyleSheet = serializedObject.FindProperty("m_defaultStyleSheet");
            m_PropColorGradientPresetsPath = serializedObject.FindProperty("m_defaultColorGradientPresetsPath");

            m_List = new ReorderableList(serializedObject, serializedObject.FindProperty("m_fallbackFontAssets"), true, true, true, true);

            m_List.drawElementCallback = (rect, index, isActive, isFocused) =>
            {
                var element = m_List.serializedProperty.GetArrayElementAtIndex(index);
                rect.y += 2;
                EditorGUI.PropertyField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), element, GUIContent.none);
            };

            m_List.drawHeaderCallback = rect =>
            {
                EditorGUI.LabelField(rect, Styles.fallbackFontAssetsListLabel);
            };

            m_PropMatchMaterialPreset = serializedObject.FindProperty("m_matchMaterialPreset");

            m_PropWordWrapping = serializedObject.FindProperty("m_enableWordWrapping");
            m_PropKerning = serializedObject.FindProperty("m_enableKerning");
            m_PropExtraPadding = serializedObject.FindProperty("m_enableExtraPadding");
            m_PropTintAllSprites = serializedObject.FindProperty("m_enableTintAllSprites");
            m_PropParseEscapeCharacters = serializedObject.FindProperty("m_enableParseEscapeCharacters");
            m_PropMissingGlyphCharacter = serializedObject.FindProperty("m_missingGlyphCharacter");

            m_PropWarningsDisabled = serializedObject.FindProperty("m_warningsDisabled");

            m_GetFontFeaturesAtRuntime = serializedObject.FindProperty("m_GetFontFeaturesAtRuntime");

            m_PropLeadingCharacters = serializedObject.FindProperty("m_leadingCharacters");
            m_PropFollowingCharacters = serializedObject.FindProperty("m_followingCharacters");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            float labelWidth = EditorGUIUtility.labelWidth;
            float fieldWidth = EditorGUIUtility.fieldWidth;

            // TextMeshPro Font Info Panel
            EditorGUI.indentLevel = 0;

            // FONT ASSET
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.defaultFontAssetLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropFontAsset, Styles.defaultFontAssetLabel);
            EditorGUILayout.PropertyField(m_PropDefaultFontAssetPath, Styles.defaultFontAssetPathLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // FALLBACK FONT ASSETs
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.fallbackFontAssetsLabel, EditorStyles.boldLabel);
            m_List.DoLayoutList();

            GUILayout.Label(Styles.fallbackMaterialSettingsLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropMatchMaterialPreset, Styles.matchMaterialPresetLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // MISSING GLYPHS
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.dynamicFontSystemSettingsLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_GetFontFeaturesAtRuntime, Styles.getFontFeaturesAtRuntime);
            EditorGUILayout.PropertyField(m_PropMissingGlyphCharacter, Styles.missingGlyphLabel);
            EditorGUILayout.PropertyField(m_PropWarningsDisabled, Styles.disableWarningsLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // TEXT OBJECT DEFAULT PROPERTIES
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.containerDefaultSettingsLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;

            EditorGUILayout.PropertyField(m_PropDefaultTextMeshProTextContainerSize, Styles.textMeshProLabel);
            EditorGUILayout.PropertyField(m_PropDefaultTextMeshProUITextContainerSize, Styles.textMeshProUiLabel);
            EditorGUILayout.PropertyField(m_PropEnableRaycastTarget, Styles.enableRaycastTarget);
            EditorGUILayout.PropertyField(m_PropAutoSizeTextContainer, Styles.autoSizeContainerLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();

            GUILayout.Label(Styles.textComponentDefaultSettingsLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropDefaultFontSize, Styles.defaultFontSize);

            EditorGUILayout.BeginHorizontal();
            {
                EditorGUILayout.PrefixLabel(Styles.autoSizeRatioLabel);
                EditorGUIUtility.labelWidth = 32;
                EditorGUIUtility.fieldWidth = 10;

                EditorGUI.indentLevel = 0;
                EditorGUILayout.PropertyField(m_PropDefaultAutoSizeMinRatio, Styles.minLabel);
                EditorGUILayout.PropertyField(m_PropDefaultAutoSizeMaxRatio, Styles.maxLabel);
                EditorGUI.indentLevel = 1;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUIUtility.fieldWidth = fieldWidth;

            EditorGUILayout.PropertyField(m_PropWordWrapping, Styles.wordWrappingLabel);
            EditorGUILayout.PropertyField(m_PropKerning, Styles.kerningLabel);

            EditorGUILayout.PropertyField(m_PropExtraPadding, Styles.extraPaddingLabel);
            EditorGUILayout.PropertyField(m_PropTintAllSprites, Styles.tintAllSpritesLabel);

            EditorGUILayout.PropertyField(m_PropParseEscapeCharacters, Styles.parseEscapeCharactersLabel);

            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // SPRITE ASSET
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.defaultSpriteAssetLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropSpriteAsset, Styles.defaultSpriteAssetLabel);
            EditorGUILayout.PropertyField(m_PropEnableEmojiSupport, Styles.enableEmojiSupportLabel);
            EditorGUILayout.PropertyField(m_PropSpriteAssetPath, Styles.spriteAssetsPathLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // STYLE SHEET
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.defaultStyleSheetLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_PropStyleSheet, Styles.defaultStyleSheetLabel);
            if (EditorGUI.EndChangeCheck())
            {
                serializedObject.ApplyModifiedProperties();
                TMP_StyleSheet.UpdateStyleSheet();
            }
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // COLOR GRADIENT PRESETS
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.colorGradientPresetsLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropColorGradientPresetsPath, Styles.colorGradientsPathLabel);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            // LINE BREAKING RULE
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.Label(Styles.lineBreakingLabel, EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            EditorGUILayout.PropertyField(m_PropLeadingCharacters);
            EditorGUILayout.PropertyField(m_PropFollowingCharacters);
            EditorGUI.indentLevel = 0;

            EditorGUILayout.Space();
            EditorGUILayout.EndVertical();

            if (serializedObject.ApplyModifiedProperties())
            {
                EditorUtility.SetDirty(target);
                TMPro_EventManager.ON_TMP_SETTINGS_CHANGED();
            }
        }
    }

#if UNITY_2018_3_OR_NEWER
    class TMP_ResourceImporterProvider : SettingsProvider
    {
        TMP_PackageResourceImporter m_ResourceImporter;

        public TMP_ResourceImporterProvider()
            : base("Project/TextMesh Pro", SettingsScope.Project)
        {
        }

        public override void OnGUI(string searchContext)
        {
            // Lazy creation that supports domain reload
            if (m_ResourceImporter == null)
                m_ResourceImporter = new TMP_PackageResourceImporter();

            m_ResourceImporter.OnGUI();
        }

        public override void OnDeactivate()
        {
            if (m_ResourceImporter != null)
                m_ResourceImporter.OnDestroy();
        }

        static UnityEngine.Object GetTMPSettings()
        {
            return Resources.Load<TMP_Settings>("TMP Settings");
        }

        [SettingsProviderGroup]
        static SettingsProvider[] CreateTMPSettingsProvider()
        {
            var providers = new List<SettingsProvider> { new TMP_ResourceImporterProvider() };

            if (GetTMPSettings() != null)
            {
                var provider = new AssetSettingsProvider("Project/TextMesh Pro/Settings", GetTMPSettings);
                provider.PopulateSearchKeywordsFromGUIContentProperties<TMP_SettingsEditor.Styles>();
                providers.Add(provider);
            }

            return providers.ToArray();
        }
    }
#endif
}
