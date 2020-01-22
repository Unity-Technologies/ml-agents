using UnityEngine;
using UnityEditor;
using UnityEditorInternal;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;


namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(TMP_FontWeightPair))]
    public class FontWeightDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            SerializedProperty prop_regular = property.FindPropertyRelative("regularTypeface");
            SerializedProperty prop_italic = property.FindPropertyRelative("italicTypeface");

            float width = position.width;

            position.width = EditorGUIUtility.labelWidth;
            EditorGUI.LabelField(position, label);

            int oldIndent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;

            // NORMAL TYPEFACE
            if (label.text[0] == '4') GUI.enabled = false;
            position.x += position.width; position.width = (width - position.width) / 2;
            EditorGUI.PropertyField(position, prop_regular, GUIContent.none);

            // ITALIC TYPEFACE
            GUI.enabled = true;
            position.x += position.width;
            EditorGUI.PropertyField(position, prop_italic, GUIContent.none);

            EditorGUI.indentLevel = oldIndent;
        }
    }

    [CustomEditor(typeof(TMP_FontAsset))]
    public class TMP_FontAssetEditor : Editor
    {
        private struct UI_PanelState
        {
            public static bool faceInfoPanel = true;
            public static bool generationSettingsPanel = true;
            public static bool fontAtlasInfoPanel = true;
            public static bool fontWeightPanel = true;
            public static bool fallbackFontAssetPanel = true;
            public static bool glyphTablePanel = false;
            public static bool characterTablePanel = false;
            public static bool fontFeatureTablePanel = false;
        }

        private struct AtlasSettings
        {
            public GlyphRenderMode glyphRenderMode;
            public int pointSize;
            public int padding;
            public int atlasWidth;
            public int atlasHeight;
        }

        /// <summary>
        /// Material used to display SDF glyphs in the Character and Glyph tables.
        /// </summary>
        internal static Material internalSDFMaterial
        {
            get
            {
                if (s_InternalSDFMaterial == null)
                {
                    Shader shader = Shader.Find("Hidden/TextMeshPro/Internal/Distance Field SSD");

                    if (shader != null)
                        s_InternalSDFMaterial = new Material(shader);
                }

                return s_InternalSDFMaterial;
            }
        }
        static Material s_InternalSDFMaterial;

        /// <summary>
        /// Material used to display Bitmap glyphs in the Character and Glyph tables.
        /// </summary>
        internal static Material internalBitmapMaterial
        {
            get
            {
                if (s_InternalBitmapMaterial == null)
                {
                    Shader shader = Shader.Find("Hidden/Internal-GUITextureClipText");

                    if (shader != null)
                        s_InternalBitmapMaterial = new Material(shader);
                }

                return s_InternalBitmapMaterial;
            }
        }
        static Material s_InternalBitmapMaterial;

        private static string[] s_UiStateLabel = new string[] { "<i>(Click to collapse)</i> ", "<i>(Click to expand)</i> " };
        private GUIContent[] m_AtlasResolutionLabels = { new GUIContent("8"), new GUIContent("16"), new GUIContent("32"), new GUIContent("64"), new GUIContent("128"), new GUIContent("256"), new GUIContent("512"), new GUIContent("1024"), new GUIContent("2048"), new GUIContent("4096"), new GUIContent("8192") };
        private int[] m_AtlasResolutions = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

        private struct Warning
        {
            public bool isEnabled;
            public double expirationTime;
        }

        private int m_CurrentGlyphPage = 0;
        private int m_CurrentCharacterPage = 0;
        private int m_CurrentKerningPage = 0;

        private int m_SelectedGlyphRecord = -1;
        private int m_SelectedCharacterRecord = -1;
        private int m_SelectedAdjustmentRecord = -1;

        private string m_dstGlyphID;
        private string m_dstUnicode;
        private const string k_placeholderUnicodeHex = "<i>New Unicode (Hex)</i>";
        private string m_unicodeHexLabel = k_placeholderUnicodeHex;
        private const string k_placeholderGlyphID = "<i>New Glyph ID</i>";
        private string m_GlyphIDLabel = k_placeholderGlyphID;

        private Warning m_AddGlyphWarning;
        private Warning m_AddCharacterWarning;
        private bool m_DisplayDestructiveChangeWarning;
        private AtlasSettings m_AtlasSettings;
        private bool m_MaterialPresetsRequireUpdate;

        private string m_GlyphSearchPattern;
        private List<int> m_GlyphSearchList;

        private string m_CharacterSearchPattern;
        private List<int> m_CharacterSearchList;

        private string m_KerningTableSearchPattern;
        private List<int> m_KerningTableSearchList;
        
        private bool m_isSearchDirty;

        private const string k_UndoRedo = "UndoRedoPerformed";

        private SerializedProperty m_AtlasPopulationMode_prop;
        private SerializedProperty font_atlas_prop;
        private SerializedProperty font_material_prop;

        private SerializedProperty m_AtlasRenderMode_prop;
        private SerializedProperty m_SamplingPointSize_prop;
        private SerializedProperty m_AtlasPadding_prop;
        private SerializedProperty m_AtlasWidth_prop;
        private SerializedProperty m_AtlasHeight_prop;

        private SerializedProperty fontWeights_prop;

        //private SerializedProperty fallbackFontAssets_prop;
        private ReorderableList m_list;

        private SerializedProperty font_normalStyle_prop;
        private SerializedProperty font_normalSpacing_prop;

        private SerializedProperty font_boldStyle_prop;
        private SerializedProperty font_boldSpacing_prop;

        private SerializedProperty font_italicStyle_prop;
        private SerializedProperty font_tabSize_prop;

        private SerializedProperty m_FaceInfo_prop;
        private SerializedProperty m_GlyphTable_prop;
        private SerializedProperty m_CharacterTable_prop;

        private TMP_FontFeatureTable m_FontFeatureTable;
        private SerializedProperty m_FontFeatureTable_prop;
        private SerializedProperty m_GlyphPairAdjustmentRecords_prop;

        private TMP_SerializedPropertyHolder m_SerializedPropertyHolder;
        private SerializedProperty m_EmptyGlyphPairAdjustmentRecord_prop;

        private TMP_FontAsset m_fontAsset;

        private Material[] m_materialPresets;

        private bool isAssetDirty = false;

        private int errorCode;

        private System.DateTime timeStamp;


        public void OnEnable()
        {
            m_FaceInfo_prop = serializedObject.FindProperty("m_FaceInfo");

            font_atlas_prop = serializedObject.FindProperty("m_AtlasTextures").GetArrayElementAtIndex(0);
            font_material_prop = serializedObject.FindProperty("material");

            m_AtlasPopulationMode_prop = serializedObject.FindProperty("m_AtlasPopulationMode");
            m_AtlasRenderMode_prop = serializedObject.FindProperty("m_AtlasRenderMode");
            m_SamplingPointSize_prop = m_FaceInfo_prop.FindPropertyRelative("m_PointSize");
            m_AtlasPadding_prop = serializedObject.FindProperty("m_AtlasPadding");
            m_AtlasWidth_prop = serializedObject.FindProperty("m_AtlasWidth");
            m_AtlasHeight_prop = serializedObject.FindProperty("m_AtlasHeight");

            fontWeights_prop = serializedObject.FindProperty("m_FontWeightTable");

            m_list = new ReorderableList(serializedObject, serializedObject.FindProperty("m_FallbackFontAssetTable"), true, true, true, true);

            m_list.drawElementCallback = (Rect rect, int index, bool isActive, bool isFocused) =>
            {
                var element = m_list.serializedProperty.GetArrayElementAtIndex(index);
                rect.y += 2;
                EditorGUI.PropertyField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), element, GUIContent.none);
            };

            m_list.drawHeaderCallback = rect =>
            {
                EditorGUI.LabelField(rect, "Fallback List");
            };

            // Clean up fallback list in the event if contains null elements.
            CleanFallbackFontAssetTable();

            font_normalStyle_prop = serializedObject.FindProperty("normalStyle");
            font_normalSpacing_prop = serializedObject.FindProperty("normalSpacingOffset");

            font_boldStyle_prop = serializedObject.FindProperty("boldStyle");
            font_boldSpacing_prop = serializedObject.FindProperty("boldSpacing");

            font_italicStyle_prop = serializedObject.FindProperty("italicStyle");
            font_tabSize_prop = serializedObject.FindProperty("tabSize");

            m_CharacterTable_prop = serializedObject.FindProperty("m_CharacterTable");
            m_GlyphTable_prop = serializedObject.FindProperty("m_GlyphTable");

            m_FontFeatureTable_prop = serializedObject.FindProperty("m_FontFeatureTable");
            m_GlyphPairAdjustmentRecords_prop = m_FontFeatureTable_prop.FindPropertyRelative("m_GlyphPairAdjustmentRecords");

            m_fontAsset = target as TMP_FontAsset;
            m_FontFeatureTable = m_fontAsset.fontFeatureTable;

            // Upgrade Font Feature Table if necessary
            if (m_fontAsset.m_KerningTable != null && m_fontAsset.m_KerningTable.kerningPairs != null && m_fontAsset.m_KerningTable.kerningPairs.Count > 0)
                m_fontAsset.ReadFontAssetDefinition();

            // Create serialized object to allow us to use a serialized property of an empty kerning pair.
            m_SerializedPropertyHolder = CreateInstance<TMP_SerializedPropertyHolder>();
            m_SerializedPropertyHolder.fontAsset = m_fontAsset;
            SerializedObject internalSerializedObject = new SerializedObject(m_SerializedPropertyHolder);
            m_EmptyGlyphPairAdjustmentRecord_prop = internalSerializedObject.FindProperty("glyphPairAdjustmentRecord");

            m_materialPresets = TMP_EditorUtility.FindMaterialReferences(m_fontAsset);

            m_GlyphSearchList = new List<int>();
            m_KerningTableSearchList = new List<int>();
        }


        public void OnDisable()
        {
            // Revert changes if user closes or changes selection without having made a choice.
            if (m_DisplayDestructiveChangeWarning)
            {
                m_DisplayDestructiveChangeWarning = false;
                RestoreAtlasGenerationSettings();
                GUIUtility.keyboardControl = 0;

                serializedObject.ApplyModifiedProperties();
            }
        }


        public override void OnInspectorGUI()
        {
            //Debug.Log("OnInspectorGUI Called.");

            Event currentEvent = Event.current;

            serializedObject.Update();

            Rect rect = EditorGUILayout.GetControlRect(false, 24);
            float labelWidth = EditorGUIUtility.labelWidth;
            float fieldWidth = EditorGUIUtility.fieldWidth;

            // FACE INFO PANEL
            #region Face info
            GUI.Label(rect, new GUIContent("<b>Face Info</b> - v" + m_fontAsset.version), TMP_UIStyleManager.sectionHeader);

            rect.x += rect.width - 132f;
            rect.y += 2;
            rect.width = 130f;
            rect.height = 18f;
            if (GUI.Button(rect, new GUIContent("Update Atlas Texture")))
            {
                TMPro_FontAssetCreatorWindow.ShowFontAtlasCreatorWindow(target as TMP_FontAsset);
            }

            EditorGUI.indentLevel = 1;
            GUI.enabled = false; // Lock UI

            // TODO : Consider creating a property drawer for these.
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_FamilyName"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_StyleName"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_PointSize"));

            GUI.enabled = true;

            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_Scale"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_LineHeight"));

            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_AscentLine"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_CapLine"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_MeanLine"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_Baseline"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_DescentLine"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_UnderlineOffset"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_UnderlineThickness"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_StrikethroughOffset"));
            //EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("strikethroughThickness"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_SuperscriptOffset"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_SuperscriptSize"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_SubscriptOffset"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_SubscriptSize"));
            EditorGUILayout.PropertyField(m_FaceInfo_prop.FindPropertyRelative("m_TabWidth"));
            // TODO : Add clamping for some of these values.
            //subSize_prop.floatValue = Mathf.Clamp(subSize_prop.floatValue, 0.25f, 1f);

            EditorGUILayout.Space();
            #endregion

            // GENERATION SETTINGS
            #region Generation Settings
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Generation Settings</b>"), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.generationSettingsPanel = !UI_PanelState.generationSettingsPanel;

            GUI.Label(rect, (UI_PanelState.generationSettingsPanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.generationSettingsPanel)
            {
                EditorGUI.indentLevel = 1;

                EditorGUI.BeginChangeCheck();
                Font sourceFont = (Font)EditorGUILayout.ObjectField("Source Font File", m_fontAsset.m_SourceFontFile_EditorRef, typeof(Font), false);
                if (EditorGUI.EndChangeCheck())
                {
                    string guid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(sourceFont));
                    m_fontAsset.m_SourceFontFileGUID = guid;
                    m_fontAsset.m_SourceFontFile_EditorRef = sourceFont;
                }

                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(m_AtlasPopulationMode_prop, new GUIContent("Atlas Population Mode"));
                if (EditorGUI.EndChangeCheck())
                {
                    serializedObject.ApplyModifiedProperties();

                    bool isDatabaseRefreshRequired = false;

                    if (m_AtlasPopulationMode_prop.intValue == 0)
                    {
                        m_fontAsset.sourceFontFile = null;

                        // Set atlas textures to non readable.
                        //for (int i = 0; i < m_fontAsset.atlasTextures.Length; i++)
                        //{
                        //    Texture2D tex = m_fontAsset.atlasTextures[i];

                        //    if (tex != null && tex.isReadable)
                        //    {
                        //        string texPath = AssetDatabase.GetAssetPath(tex);
                        //        var texImporter = AssetImporter.GetAtPath(texPath) as TextureImporter;
                        //        if (texImporter != null)
                        //        {
                        //            texImporter.isReadable = false;
                        //            AssetDatabase.ImportAsset(texPath);
                        //            isDatabaseRefreshRequired = true;
                        //        }
                        //    }
                        //}

                        Debug.Log("Atlas Population mode set to [Static].");
                    }
                    else if (m_AtlasPopulationMode_prop.intValue == 1)
                    {
                        if (m_fontAsset.m_SourceFontFile_EditorRef.dynamic == false)
                        {
                            Debug.LogWarning("Please set the [" + m_fontAsset.name + "] font to dynamic mode as this is required for Dynamic SDF support.", m_fontAsset.m_SourceFontFile_EditorRef);
                            m_AtlasPopulationMode_prop.intValue = 0;

                            serializedObject.ApplyModifiedProperties();
                        }
                        else
                        {
                            m_fontAsset.sourceFontFile = m_fontAsset.m_SourceFontFile_EditorRef;

                            /*
                            // Set atlas textures to non readable.
                            for (int i = 0; i < m_fontAsset.atlasTextures.Length; i++)
                            {
                                Texture2D tex = m_fontAsset.atlasTextures[i];

                                if (tex != null && tex.isReadable == false)
                                {
                                    string texPath = AssetDatabase.GetAssetPath(tex.GetInstanceID());
                                    Object[] paths = AssetDatabase.LoadAllAssetsAtPath(texPath);
                                    var texImporter = AssetImporter.GetAtPath(texPath) as TextureImporter;
                                    if (texImporter != null)
                                    {
                                        texImporter.isReadable = true;
                                        AssetDatabase.ImportAsset(texPath);
                                        isDatabaseRefreshRequired = true;
                                    }
                                }
                            }
                            */
                            Debug.Log("Atlas Population mode set to [Dynamic].");
                        }
                    }

                    if (isDatabaseRefreshRequired)
                        AssetDatabase.Refresh();

                    serializedObject.Update();
                    isAssetDirty = true;
                }

                GUI.enabled = true;
                // Save state of atlas settings
                if (m_DisplayDestructiveChangeWarning == false)
                {
                    SavedAtlasGenerationSettings();
                    //Undo.RegisterCompleteObjectUndo(m_fontAsset, "Font Asset Changes");
                }

                EditorGUI.BeginChangeCheck();
                // TODO: Switch shaders depending on GlyphRenderMode.
                EditorGUILayout.PropertyField(m_AtlasRenderMode_prop);
                EditorGUILayout.PropertyField(m_SamplingPointSize_prop, new GUIContent("Sampling Point Size"));
                if (EditorGUI.EndChangeCheck())
                {
                    m_DisplayDestructiveChangeWarning = true;
                }

                // Changes to these properties require updating Material Presets for this font asset.
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(m_AtlasPadding_prop, new GUIContent("Padding"));
                EditorGUILayout.IntPopup(m_AtlasWidth_prop, m_AtlasResolutionLabels, m_AtlasResolutions, new GUIContent("Atlas Width"));
                EditorGUILayout.IntPopup(m_AtlasHeight_prop, m_AtlasResolutionLabels, m_AtlasResolutions, new GUIContent("Atlas Height"));
                if (EditorGUI.EndChangeCheck())
                {
                    m_MaterialPresetsRequireUpdate = true;
                    m_DisplayDestructiveChangeWarning = true;
                }

                if (m_DisplayDestructiveChangeWarning)
                {
                    // These changes are destructive on the font asset
                    rect = EditorGUILayout.GetControlRect(false, 60);
                    rect.x += 15;
                    rect.width -= 15;
                    EditorGUI.HelpBox(rect, "Changing these settings will clear the font asset's character, glyph and texture data.", MessageType.Warning);

                    if (GUI.Button(new Rect(rect.width - 140, rect.y + 36, 80, 18), new GUIContent("Apply")))
                    {
                        m_DisplayDestructiveChangeWarning = false;

                        // Update face info is sampling point size was changed.
                        if (m_AtlasSettings.pointSize != m_SamplingPointSize_prop.intValue)
                        {
                            FontEngine.LoadFontFace(m_fontAsset.sourceFontFile, m_SamplingPointSize_prop.intValue);
                            m_fontAsset.faceInfo = FontEngine.GetFaceInfo();
                        }

                        // Update material
                        m_fontAsset.material.SetFloat(ShaderUtilities.ID_TextureWidth, m_AtlasWidth_prop.intValue);
                        m_fontAsset.material.SetFloat(ShaderUtilities.ID_TextureHeight, m_AtlasHeight_prop.intValue);
                        m_fontAsset.material.SetFloat(ShaderUtilities.ID_GradientScale, m_AtlasPadding_prop.intValue + 1);

                        // Update material presets if any of the relevant properties have been changed.
                        if (m_MaterialPresetsRequireUpdate)
                        {
                            m_MaterialPresetsRequireUpdate = false;

                            Material[] materialPresets = TMP_EditorUtility.FindMaterialReferences(m_fontAsset);
                            for (int i = 0; i < materialPresets.Length; i++)
                            {
                                Material mat = materialPresets[i];

                                mat.SetFloat(ShaderUtilities.ID_TextureWidth, m_AtlasWidth_prop.intValue);
                                mat.SetFloat(ShaderUtilities.ID_TextureHeight, m_AtlasHeight_prop.intValue);
                                mat.SetFloat(ShaderUtilities.ID_GradientScale, m_AtlasPadding_prop.intValue + 1);
                            }
                        }

                        m_fontAsset.ClearFontAssetData();
                        GUIUtility.keyboardControl = 0;
                        isAssetDirty = true;

                        // Update Font Asset Creation Settings to reflect new changes.
                        UpdateFontAssetCreationSettings();

                        // TODO: Clear undo buffers.
                        //Undo.ClearUndo(m_fontAsset);
                    }

                    if (GUI.Button(new Rect(rect.width - 56, rect.y + 36, 80, 18), new GUIContent("Revert")))
                    {
                        m_DisplayDestructiveChangeWarning = false;
                        RestoreAtlasGenerationSettings();
                        GUIUtility.keyboardControl = 0;

                        // TODO: Clear undo buffers.
                        //Undo.ClearUndo(m_fontAsset);
                    }
                }
                EditorGUILayout.Space();
            }
            #endregion

            // ATLAS & MATERIAL PANEL
            #region Atlas & Material
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Atlas & Material</b>"), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.fontAtlasInfoPanel = !UI_PanelState.fontAtlasInfoPanel;

            GUI.Label(rect, (UI_PanelState.fontAtlasInfoPanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.fontAtlasInfoPanel)
            {
                EditorGUI.indentLevel = 1;

                GUI.enabled = false;
                EditorGUILayout.PropertyField(font_atlas_prop, new GUIContent("Font Atlas"));
                EditorGUILayout.PropertyField(font_material_prop, new GUIContent("Font Material"));
                GUI.enabled = true;
                EditorGUILayout.Space();
            }
            #endregion

            string evt_cmd = Event.current.commandName; // Get Current Event CommandName to check for Undo Events

            // FONT WEIGHT PANEL
            #region Font Weights
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Font Weights</b>", "The Font Assets that will be used for different font weights and the settings used to simulate a typeface when no asset is available."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.fontWeightPanel = !UI_PanelState.fontWeightPanel;

            GUI.Label(rect, (UI_PanelState.fontWeightPanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.fontWeightPanel)
            {
                EditorGUIUtility.labelWidth *= 0.75f;
                EditorGUIUtility.fieldWidth *= 0.25f;

                EditorGUILayout.BeginVertical();
                EditorGUI.indentLevel = 1;
                rect = EditorGUILayout.GetControlRect(true);
                rect.x += EditorGUIUtility.labelWidth;
                rect.width = (rect.width - EditorGUIUtility.labelWidth) / 2f;
                GUI.Label(rect, "Regular Tyepface", EditorStyles.label);
                rect.x += rect.width;
                GUI.Label(rect, "Italic Typeface", EditorStyles.label);
                
                EditorGUI.indentLevel = 1;

                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(1), new GUIContent("100 - Thin"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(2), new GUIContent("200 - Extra-Light"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(3), new GUIContent("300 - Light"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(4), new GUIContent("400 - Regular"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(5), new GUIContent("500 - Medium"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(6), new GUIContent("600 - Semi-Bold"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(7), new GUIContent("700 - Bold"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(8), new GUIContent("800 - Heavy"));
                EditorGUILayout.PropertyField(fontWeights_prop.GetArrayElementAtIndex(9), new GUIContent("900 - Black"));

                EditorGUILayout.EndVertical();

                EditorGUILayout.Space();

                EditorGUILayout.BeginVertical();

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(font_normalStyle_prop, new GUIContent("Normal Weight"));
                font_normalStyle_prop.floatValue = Mathf.Clamp(font_normalStyle_prop.floatValue, -3.0f, 3.0f);
                if (GUI.changed || evt_cmd == k_UndoRedo)
                {
                    GUI.changed = false;

                    // Modify the material property on matching material presets.
                    for (int i = 0; i < m_materialPresets.Length; i++)
                        m_materialPresets[i].SetFloat("_WeightNormal", font_normalStyle_prop.floatValue);
                }

                EditorGUILayout.PropertyField(font_boldStyle_prop, new GUIContent("Bold Weight"));
                font_boldStyle_prop.floatValue = Mathf.Clamp(font_boldStyle_prop.floatValue, -3.0f, 3.0f);
                if (GUI.changed || evt_cmd == k_UndoRedo)
                {
                    GUI.changed = false;

                    // Modify the material property on matching material presets.
                    for (int i = 0; i < m_materialPresets.Length; i++)
                        m_materialPresets[i].SetFloat("_WeightBold", font_boldStyle_prop.floatValue);
                }
                EditorGUILayout.EndHorizontal();
                
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(font_normalSpacing_prop, new GUIContent("Spacing Offset"));
                font_normalSpacing_prop.floatValue = Mathf.Clamp(font_normalSpacing_prop.floatValue, -100, 100);
                if (GUI.changed || evt_cmd == k_UndoRedo)
                {
                    GUI.changed = false;
                }

                EditorGUILayout.PropertyField(font_boldSpacing_prop, new GUIContent("Bold Spacing"));
                font_boldSpacing_prop.floatValue = Mathf.Clamp(font_boldSpacing_prop.floatValue, 0, 100);
                if (GUI.changed || evt_cmd == k_UndoRedo)
                {
                    GUI.changed = false;
                }
                EditorGUILayout.EndHorizontal();
                
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.PropertyField(font_italicStyle_prop, new GUIContent("Italic Style"));
                font_italicStyle_prop.intValue = Mathf.Clamp(font_italicStyle_prop.intValue, 15, 60);
                
                EditorGUILayout.PropertyField(font_tabSize_prop, new GUIContent("Tab Multiple"));
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.EndVertical();
                EditorGUILayout.Space();
            }

            EditorGUIUtility.labelWidth = 0;
            EditorGUIUtility.fieldWidth = 0;
            #endregion

            // FALLBACK FONT ASSETS
            #region Fallback Font Asset
            rect = EditorGUILayout.GetControlRect(false, 24);
            EditorGUI.indentLevel = 0;
            if (GUI.Button(rect, new GUIContent("<b>Fallback Font Assets</b>", "Select the Font Assets that will be searched and used as fallback when characters are missing from this font asset."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.fallbackFontAssetPanel = !UI_PanelState.fallbackFontAssetPanel;

            GUI.Label(rect, (UI_PanelState.fallbackFontAssetPanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.fallbackFontAssetPanel)
            {
                EditorGUIUtility.labelWidth = 120;
                EditorGUI.indentLevel = 0;

                m_list.DoLayoutList();
                EditorGUILayout.Space();
            }
            #endregion

            // CHARACTER TABLE TABLE
            #region Character Table
            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUIUtility.fieldWidth = fieldWidth;
            EditorGUI.indentLevel = 0;
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Character Table</b>", "List of characters contained in this font asset."), TMP_UIStyleManager.sectionHeader)) 
                UI_PanelState.characterTablePanel = !UI_PanelState.characterTablePanel;

            GUI.Label(rect, (UI_PanelState.characterTablePanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.characterTablePanel)
            {
                int arraySize = m_CharacterTable_prop.arraySize;
                int itemsPerPage = 15;

                // Display Glyph Management Tools
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    // Search Bar implementation
                    #region DISPLAY SEARCH BAR
                    EditorGUILayout.BeginHorizontal();
                    {
                        EditorGUIUtility.labelWidth = 130f;
                        EditorGUI.BeginChangeCheck();
                        string searchPattern = EditorGUILayout.TextField("Character Search", m_CharacterSearchPattern, "SearchTextField");
                        if (EditorGUI.EndChangeCheck() || m_isSearchDirty)
                        {
                            if (string.IsNullOrEmpty(searchPattern) == false)
                            {
                                m_CharacterSearchPattern = searchPattern;

                                // Search Character Table for potential matches
                                SearchCharacterTable (m_CharacterSearchPattern, ref m_CharacterSearchList);
                            }
                            else
                                m_CharacterSearchPattern = null;

                            m_isSearchDirty = false;
                        }

                        string styleName = string.IsNullOrEmpty(m_CharacterSearchPattern) ? "SearchCancelButtonEmpty" : "SearchCancelButton";
                        if (GUILayout.Button(GUIContent.none, styleName))
                        {
                            GUIUtility.keyboardControl = 0;
                            m_CharacterSearchPattern = string.Empty;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                    #endregion

                    // Display Page Navigation
                    if (!string.IsNullOrEmpty(m_CharacterSearchPattern))
                        arraySize = m_CharacterSearchList.Count;

                    DisplayPageNavigation(ref m_CurrentCharacterPage, arraySize, itemsPerPage);
                }
                EditorGUILayout.EndVertical();

                // Display Character Table Elements
                if (arraySize > 0)
                {
                    // Display each character entry using the CharacterPropertyDrawer.
                    for (int i = itemsPerPage * m_CurrentCharacterPage; i < arraySize && i < itemsPerPage * (m_CurrentCharacterPage + 1); i++)
                    {
                        // Define the start of the selection region of the element.
                        Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        int elementIndex = i;
                        if (!string.IsNullOrEmpty(m_CharacterSearchPattern))
                            elementIndex = m_CharacterSearchList[i];

                        SerializedProperty characterProperty = m_CharacterTable_prop.GetArrayElementAtIndex(elementIndex);

                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);

                        EditorGUI.BeginDisabledGroup(i != m_SelectedCharacterRecord);
                        {
                            EditorGUILayout.PropertyField(characterProperty);
                        }
                        EditorGUI.EndDisabledGroup();

                        EditorGUILayout.EndVertical();

                        // Define the end of the selection region of the element.
                        Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        // Check for Item selection
                        Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                        if (DoSelectionCheck(selectionArea))
                        {
                            if (m_SelectedCharacterRecord == i)
                                m_SelectedCharacterRecord = -1;
                            else
                            {
                                m_SelectedCharacterRecord = i;
                                m_AddCharacterWarning.isEnabled = false;
                                m_unicodeHexLabel = k_placeholderUnicodeHex;
                                GUIUtility.keyboardControl = 0;
                            }
                        }

                        // Draw Selection Highlight and Glyph Options
                        if (m_SelectedCharacterRecord == i)
                        {
                            TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));

                            // Draw Glyph management options
                            Rect controlRect = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight * 1f);
                            float optionAreaWidth = controlRect.width * 0.6f;
                            float btnWidth = optionAreaWidth / 3;

                            Rect position = new Rect(controlRect.x + controlRect.width * .4f, controlRect.y, btnWidth, controlRect.height);

                            // Copy Selected Glyph to Target Glyph ID
                            GUI.enabled = !string.IsNullOrEmpty(m_dstUnicode);
                            if (GUI.Button(position, new GUIContent("Copy to")))
                            {
                                GUIUtility.keyboardControl = 0;

                                // Convert Hex Value to Decimal
                                int dstGlyphID = TMP_TextUtilities.StringHexToInt(m_dstUnicode);

                                //Add new glyph at target Unicode hex id.
                                if (!AddNewCharacter(elementIndex, dstGlyphID))
                                {
                                    m_AddCharacterWarning.isEnabled = true;
                                    m_AddCharacterWarning.expirationTime = EditorApplication.timeSinceStartup + 1;
                                }

                                m_dstUnicode = string.Empty;
                                m_isSearchDirty = true;

                                TMPro_EventManager.ON_FONT_PROPERTY_CHANGED(true, m_fontAsset);
                            }

                            // Target Glyph ID
                            GUI.enabled = true;
                            position.x += btnWidth;

                            GUI.SetNextControlName("CharacterID_Input");
                            m_dstUnicode = EditorGUI.TextField(position, m_dstUnicode);

                            // Placeholder text
                            EditorGUI.LabelField(position, new GUIContent(m_unicodeHexLabel, "The Unicode (Hex) ID of the duplicated Character"), TMP_UIStyleManager.label);

                            // Only filter the input when the destination glyph ID text field has focus.
                            if (GUI.GetNameOfFocusedControl() == "CharacterID_Input")
                            {
                                m_unicodeHexLabel = string.Empty;

                                //Filter out unwanted characters.
                                char chr = Event.current.character;
                                if ((chr < '0' || chr > '9') && (chr < 'a' || chr > 'f') && (chr < 'A' || chr > 'F'))
                                {
                                    Event.current.character = '\0';
                                }
                            }
                            else
                            {
                                m_unicodeHexLabel = k_placeholderUnicodeHex;
                                //m_dstUnicode = string.Empty;
                            }


                            // Remove Glyph
                            position.x += btnWidth;
                            if (GUI.Button(position, "Remove"))
                            {
                                GUIUtility.keyboardControl = 0;

                                RemoveCharacterFromList(elementIndex);

                                isAssetDirty = true;
                                m_SelectedCharacterRecord = -1;
                                m_isSearchDirty = true;
                                break;
                            }

                            if (m_AddCharacterWarning.isEnabled && EditorApplication.timeSinceStartup < m_AddCharacterWarning.expirationTime)
                            {
                                EditorGUILayout.HelpBox("The Destination Character ID already exists", MessageType.Warning);
                            }

                        }
                    }
                }

                DisplayPageNavigation(ref m_CurrentCharacterPage, arraySize, itemsPerPage);

                EditorGUILayout.Space();
            }
            #endregion

            // GLYPH TABLE
            #region Glyph Table
            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUIUtility.fieldWidth = fieldWidth;
            EditorGUI.indentLevel = 0;
            rect = EditorGUILayout.GetControlRect(false, 24);

            GUIStyle glyphPanelStyle = new GUIStyle(EditorStyles.helpBox);

            if (GUI.Button(rect, new GUIContent("<b>Glyph Table</b>", "List of glyphs contained in this font asset."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.glyphTablePanel = !UI_PanelState.glyphTablePanel;

            GUI.Label(rect, (UI_PanelState.glyphTablePanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.glyphTablePanel)
            {
                int arraySize = m_GlyphTable_prop.arraySize;
                int itemsPerPage = 15;

                // Display Glyph Management Tools
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    // Search Bar implementation
                    #region DISPLAY SEARCH BAR
                    EditorGUILayout.BeginHorizontal();
                    {
                        EditorGUIUtility.labelWidth = 130f;
                        EditorGUI.BeginChangeCheck();
                        string searchPattern = EditorGUILayout.TextField("Glyph Search", m_GlyphSearchPattern, "SearchTextField");
                        if (EditorGUI.EndChangeCheck() || m_isSearchDirty)
                        {
                            if (string.IsNullOrEmpty(searchPattern) == false)
                            {
                                m_GlyphSearchPattern = searchPattern;

                                // Search Glyph Table for potential matches
                                SearchGlyphTable(m_GlyphSearchPattern, ref m_GlyphSearchList);
                            }
                            else
                                m_GlyphSearchPattern = null;

                            m_isSearchDirty = false;
                        }

                        string styleName = string.IsNullOrEmpty(m_GlyphSearchPattern) ? "SearchCancelButtonEmpty" : "SearchCancelButton";
                        if (GUILayout.Button(GUIContent.none, styleName))
                        {
                            GUIUtility.keyboardControl = 0;
                            m_GlyphSearchPattern = string.Empty;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                    #endregion

                    // Display Page Navigation
                    if (!string.IsNullOrEmpty(m_GlyphSearchPattern))
                        arraySize = m_GlyphSearchList.Count;

                    DisplayPageNavigation(ref m_CurrentGlyphPage, arraySize, itemsPerPage);
                }
                EditorGUILayout.EndVertical();

                // Display Glyph Table Elements
                
                if (arraySize > 0)
                {
                    // Display each GlyphInfo entry using the GlyphInfo property drawer.
                    for (int i = itemsPerPage * m_CurrentGlyphPage; i < arraySize && i < itemsPerPage * (m_CurrentGlyphPage + 1); i++)
                    {
                        // Define the start of the selection region of the element.
                        Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        int elementIndex = i;
                        if (!string.IsNullOrEmpty(m_GlyphSearchPattern))
                            elementIndex = m_GlyphSearchList[i];

                        SerializedProperty glyphProperty = m_GlyphTable_prop.GetArrayElementAtIndex(elementIndex);

                        EditorGUILayout.BeginVertical(glyphPanelStyle);

                        using (new EditorGUI.DisabledScope(i != m_SelectedGlyphRecord))
                        {
                            EditorGUILayout.PropertyField(glyphProperty);
                        }

                        EditorGUILayout.EndVertical();

                        // Define the end of the selection region of the element.
                        Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        // Check for Item selection
                        Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                        if (DoSelectionCheck(selectionArea))
                        {
                            if (m_SelectedGlyphRecord == i)
                                m_SelectedGlyphRecord = -1;
                            else
                            {
                                m_SelectedGlyphRecord = i;
                                m_AddGlyphWarning.isEnabled = false;
                                m_unicodeHexLabel = k_placeholderUnicodeHex;
                                GUIUtility.keyboardControl = 0;
                            }
                        }

                        // Draw Selection Highlight and Glyph Options
                        if (m_SelectedGlyphRecord == i)
                        {
                            TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));

                            // Draw Glyph management options
                            Rect controlRect = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight * 1f);
                            float optionAreaWidth = controlRect.width * 0.6f;
                            float btnWidth = optionAreaWidth / 3;

                            Rect position = new Rect(controlRect.x + controlRect.width * .4f, controlRect.y, btnWidth, controlRect.height);

                            // Copy Selected Glyph to Target Glyph ID
                            GUI.enabled = !string.IsNullOrEmpty(m_dstGlyphID);
                            if (GUI.Button(position, new GUIContent("Copy to")))
                            {
                                GUIUtility.keyboardControl = 0;
                                int dstGlyphID;

                                // Convert Hex Value to Decimal
                                int.TryParse(m_dstGlyphID, out dstGlyphID);

                                //Add new glyph at target Unicode hex id.
                                if (!AddNewGlyph(elementIndex, dstGlyphID))
                                {
                                    m_AddGlyphWarning.isEnabled = true;
                                    m_AddGlyphWarning.expirationTime = EditorApplication.timeSinceStartup + 1;
                                }

                                m_dstGlyphID = string.Empty;
                                m_isSearchDirty = true;

                                TMPro_EventManager.ON_FONT_PROPERTY_CHANGED(true, m_fontAsset);
                            }

                            // Target Glyph ID
                            GUI.enabled = true;
                            position.x += btnWidth;

                            GUI.SetNextControlName("GlyphID_Input");
                            m_dstGlyphID = EditorGUI.TextField(position, m_dstGlyphID);

                            // Placeholder text
                            EditorGUI.LabelField(position, new GUIContent(m_GlyphIDLabel, "The Glyph ID of the duplicated Glyph"), TMP_UIStyleManager.label);

                            // Only filter the input when the destination glyph ID text field has focus.
                            if (GUI.GetNameOfFocusedControl() == "GlyphID_Input")
                            {
                                m_GlyphIDLabel = string.Empty;

                                //Filter out unwanted characters.
                                char chr = Event.current.character;
                                if ((chr < '0' || chr > '9'))
                                {
                                    Event.current.character = '\0';
                                }
                            }
                            else
                            {
                                m_GlyphIDLabel = k_placeholderGlyphID;
                                //m_dstGlyphID = string.Empty;
                            }

                            // Remove Glyph
                            position.x += btnWidth;
                            if (GUI.Button(position, "Remove"))
                            {
                                GUIUtility.keyboardControl = 0;

                                RemoveGlyphFromList(elementIndex);

                                isAssetDirty = true;
                                m_SelectedGlyphRecord = -1;
                                m_isSearchDirty = true;
                                break;
                            }

                            if (m_AddGlyphWarning.isEnabled && EditorApplication.timeSinceStartup < m_AddGlyphWarning.expirationTime)
                            {
                                EditorGUILayout.HelpBox("The Destination Glyph ID already exists", MessageType.Warning);
                            }

                        }
                    }
                }

                DisplayPageNavigation(ref m_CurrentGlyphPage, arraySize, itemsPerPage);

                EditorGUILayout.Space();
            }
            #endregion

            // FONT FEATURE TABLE
            #region Font Feature Table
            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUIUtility.fieldWidth = fieldWidth;
            EditorGUI.indentLevel = 0;
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Glyph Adjustment Table</b>", "List of glyph adjustment / advanced kerning pairs."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.fontFeatureTablePanel = !UI_PanelState.fontFeatureTablePanel;

            GUI.Label(rect, (UI_PanelState.fontFeatureTablePanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.fontFeatureTablePanel)
            {
                int arraySize = m_GlyphPairAdjustmentRecords_prop.arraySize;
                int itemsPerPage = 20;

                // Display Kerning Pair Management Tools
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    // Search Bar implementation
                    #region DISPLAY SEARCH BAR
                    EditorGUILayout.BeginHorizontal();
                    {
                        EditorGUIUtility.labelWidth = 150f;
                        EditorGUI.BeginChangeCheck();
                        string searchPattern = EditorGUILayout.TextField("Adjustment Pair Search", m_KerningTableSearchPattern, "SearchTextField");
                        if (EditorGUI.EndChangeCheck() || m_isSearchDirty)
                        {
                            if (string.IsNullOrEmpty(searchPattern) == false)
                            {
                                m_KerningTableSearchPattern = searchPattern;

                                // Search Glyph Table for potential matches
                                SearchKerningTable(m_KerningTableSearchPattern, ref m_KerningTableSearchList);
                            }
                            else
                                m_KerningTableSearchPattern = null;

                            m_isSearchDirty = false;
                        }

                        string styleName = string.IsNullOrEmpty(m_KerningTableSearchPattern) ? "SearchCancelButtonEmpty" : "SearchCancelButton";
                        if (GUILayout.Button(GUIContent.none, styleName))
                        {
                            GUIUtility.keyboardControl = 0;
                            m_KerningTableSearchPattern = string.Empty;
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                    #endregion

                    // Display Page Navigation
                    if (!string.IsNullOrEmpty(m_KerningTableSearchPattern))
                        arraySize = m_KerningTableSearchList.Count;

                    DisplayPageNavigation(ref m_CurrentKerningPage, arraySize, itemsPerPage);
                }
                EditorGUILayout.EndVertical();

                if (arraySize > 0)
                {
                    // Display each GlyphInfo entry using the GlyphInfo property drawer.
                    for (int i = itemsPerPage * m_CurrentKerningPage; i < arraySize && i < itemsPerPage * (m_CurrentKerningPage + 1); i++)
                    {
                        // Define the start of the selection region of the element.
                        Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        int elementIndex = i;
                        if (!string.IsNullOrEmpty(m_KerningTableSearchPattern))
                            elementIndex = m_KerningTableSearchList[i];

                        SerializedProperty pairAdjustmentRecordProperty = m_GlyphPairAdjustmentRecords_prop.GetArrayElementAtIndex(elementIndex);

                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);

                        using (new EditorGUI.DisabledScope(i != m_SelectedAdjustmentRecord))
                        {
                            EditorGUILayout.PropertyField(pairAdjustmentRecordProperty, new GUIContent("Selectable"));
                        }

                        EditorGUILayout.EndVertical();

                        // Define the end of the selection region of the element.
                        Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        // Check for Item selection
                        Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                        if (DoSelectionCheck(selectionArea))
                        {
                            if (m_SelectedAdjustmentRecord == i)
                            {
                                m_SelectedAdjustmentRecord = -1;
                            }
                            else
                            {
                                m_SelectedAdjustmentRecord = i;
                                GUIUtility.keyboardControl = 0;
                            }
                        }

                        // Draw Selection Highlight and Kerning Pair Options
                        if (m_SelectedAdjustmentRecord == i)
                        {
                            TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));

                            // Draw Glyph management options
                            Rect controlRect = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight * 1f);
                            float optionAreaWidth = controlRect.width;
                            float btnWidth = optionAreaWidth / 4;

                            Rect position = new Rect(controlRect.x + controlRect.width - btnWidth, controlRect.y, btnWidth, controlRect.height);

                            // Remove Kerning pair
                            GUI.enabled = true;
                            if (GUI.Button(position, "Remove"))
                            {
                                GUIUtility.keyboardControl = 0;

                                RemoveAdjustmentPairFromList(i);

                                isAssetDirty = true;
                                m_SelectedAdjustmentRecord = -1;
                                m_isSearchDirty = true;
                                break;
                            }
                        }
                    }
                }

                DisplayPageNavigation(ref m_CurrentKerningPage, arraySize, itemsPerPage);

                GUILayout.Space(5);

                // Add new kerning pair
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                {
                    EditorGUILayout.PropertyField(m_EmptyGlyphPairAdjustmentRecord_prop);
                }
                EditorGUILayout.EndVertical();

                if (GUILayout.Button("Add New Glyph Adjustment Record"))
                {
                    SerializedProperty firstAdjustmentRecordProperty = m_EmptyGlyphPairAdjustmentRecord_prop.FindPropertyRelative("m_FirstAdjustmentRecord");
                    SerializedProperty secondAdjustmentRecordProperty = m_EmptyGlyphPairAdjustmentRecord_prop.FindPropertyRelative("m_SecondAdjustmentRecord");

                    uint firstGlyphIndex = (uint)firstAdjustmentRecordProperty.FindPropertyRelative("m_GlyphIndex").intValue;
                    uint secondGlyphIndex = (uint)secondAdjustmentRecordProperty.FindPropertyRelative("m_GlyphIndex").intValue;

                    TMP_GlyphValueRecord firstValueRecord = GetValueRecord(firstAdjustmentRecordProperty.FindPropertyRelative("m_GlyphValueRecord"));
                    TMP_GlyphValueRecord secondValueRecord = GetValueRecord(secondAdjustmentRecordProperty.FindPropertyRelative("m_GlyphValueRecord"));

                    errorCode = -1;
                    long pairKey = (long)secondGlyphIndex << 32 | firstGlyphIndex;
                    if (m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.ContainsKey(pairKey) == false)
                    {
                        TMP_GlyphPairAdjustmentRecord adjustmentRecord = new TMP_GlyphPairAdjustmentRecord(new TMP_GlyphAdjustmentRecord(firstGlyphIndex, firstValueRecord), new TMP_GlyphAdjustmentRecord(secondGlyphIndex, secondValueRecord));
                        m_FontFeatureTable.m_GlyphPairAdjustmentRecords.Add(adjustmentRecord);
                        m_FontFeatureTable.m_GlyphPairAdjustmentRecordLookupDictionary.Add(pairKey, adjustmentRecord);
                        errorCode = 0;
                    }

                    TMP_Character character;

                    // Add glyphs and characters
                    uint firstCharacter = m_SerializedPropertyHolder.firstCharacter;
                    if (!m_fontAsset.characterLookupTable.ContainsKey(firstCharacter))
                        m_fontAsset.TryAddCharacterInternal(firstCharacter, out character);

                    uint secondCharacter = m_SerializedPropertyHolder.secondCharacter;
                    if (!m_fontAsset.characterLookupTable.ContainsKey(secondCharacter))
                        m_fontAsset.TryAddCharacterInternal(secondCharacter, out character);

                    // Sort Kerning Pairs & Reload Font Asset if new kerning pair was added.
                    if (errorCode != -1)
                    {
                        m_FontFeatureTable.SortGlyphPairAdjustmentRecords();
                        serializedObject.ApplyModifiedProperties();
                        isAssetDirty = true;
                        m_isSearchDirty = true;
                    }
                    else
                    {
                        timeStamp = System.DateTime.Now.AddSeconds(5);
                    }

                    // Clear Add Kerning Pair Panel
                    // TODO
                }

                if (errorCode == -1)
                {
                    GUILayout.BeginHorizontal();
                    GUILayout.FlexibleSpace();
                    GUILayout.Label("Kerning Pair already <color=#ffff00>exists!</color>", TMP_UIStyleManager.label);
                    GUILayout.FlexibleSpace();
                    GUILayout.EndHorizontal();

                    if (System.DateTime.Now > timeStamp)
                        errorCode = 0;
                }
            }
            #endregion

            if (serializedObject.ApplyModifiedProperties() || evt_cmd == k_UndoRedo || isAssetDirty)
            {
                // Delay callback until user has decided to Apply or Revert the changes.
                if (m_DisplayDestructiveChangeWarning == false)
                    TMPro_EventManager.ON_FONT_PROPERTY_CHANGED(true, m_fontAsset);

                if (m_fontAsset.m_IsFontAssetLookupTablesDirty || evt_cmd == k_UndoRedo)
                    m_fontAsset.ReadFontAssetDefinition();

                isAssetDirty = false;
                EditorUtility.SetDirty(target);
            }


            // Clear selection if mouse event was not consumed. 
            GUI.enabled = true;
            if (currentEvent.type == EventType.MouseDown && currentEvent.button == 0)
                m_SelectedAdjustmentRecord = -1;

        }

        void CleanFallbackFontAssetTable()
        {
            SerializedProperty m_FallbackFontAsseTable = serializedObject.FindProperty("m_FallbackFontAssetTable");

            bool isListDirty = false;

            int elementCount = m_FallbackFontAsseTable.arraySize;

            for (int i = 0; i < elementCount; i++)
            {
                SerializedProperty element = m_FallbackFontAsseTable.GetArrayElementAtIndex(i);
                if (element.objectReferenceValue == null)
                {
                    m_FallbackFontAsseTable.DeleteArrayElementAtIndex(i);
                    elementCount -= 1;
                    i -= 1;

                    isListDirty = true;
                }
            }

            if (isListDirty)
            {
                serializedObject.ApplyModifiedProperties();
                serializedObject.Update();
            }
        }

        void SavedAtlasGenerationSettings()
        {
            m_AtlasSettings.glyphRenderMode = (GlyphRenderMode)m_AtlasRenderMode_prop.intValue;
            m_AtlasSettings.pointSize       = m_SamplingPointSize_prop.intValue;
            m_AtlasSettings.padding         = m_AtlasPadding_prop.intValue;
            m_AtlasSettings.atlasWidth      = m_AtlasWidth_prop.intValue;
            m_AtlasSettings.atlasHeight     = m_AtlasHeight_prop.intValue;
        }

        void RestoreAtlasGenerationSettings()
        {
            m_AtlasRenderMode_prop.intValue = (int)m_AtlasSettings.glyphRenderMode;
            m_SamplingPointSize_prop.intValue = m_AtlasSettings.pointSize;
            m_AtlasPadding_prop.intValue = m_AtlasSettings.padding;
            m_AtlasWidth_prop.intValue = m_AtlasSettings.atlasWidth;
            m_AtlasHeight_prop.intValue = m_AtlasSettings.atlasHeight;
        }


        void UpdateFontAssetCreationSettings()
        {
            m_fontAsset.m_CreationSettings.pointSize = m_SamplingPointSize_prop.intValue;
            m_fontAsset.m_CreationSettings.renderMode = m_AtlasRenderMode_prop.intValue;
            m_fontAsset.m_CreationSettings.padding = m_AtlasPadding_prop.intValue;
            m_fontAsset.m_CreationSettings.atlasWidth = m_AtlasWidth_prop.intValue;
            m_fontAsset.m_CreationSettings.atlasHeight = m_AtlasHeight_prop.intValue;
        }


        void UpdateCharacterData(SerializedProperty property, int index)
        {
            TMP_Character character = m_fontAsset.characterTable[index];

            character.unicode = (uint)property.FindPropertyRelative("m_Unicode").intValue;
            character.scale = property.FindPropertyRelative("m_Scale").floatValue;

            SerializedProperty glyphProperty = property.FindPropertyRelative("m_Glyph");
            character.glyph.index = (uint)glyphProperty.FindPropertyRelative("m_Index").intValue;

            SerializedProperty glyphRectProperty = glyphProperty.FindPropertyRelative("m_GlyphRect");
            character.glyph.glyphRect = new GlyphRect(glyphRectProperty.FindPropertyRelative("m_X").intValue, glyphRectProperty.FindPropertyRelative("m_Y").intValue, glyphRectProperty.FindPropertyRelative("m_Width").intValue, glyphRectProperty.FindPropertyRelative("m_Height").intValue);

            SerializedProperty glyphMetricsProperty = glyphProperty.FindPropertyRelative("m_Metrics");
            character.glyph.metrics = new GlyphMetrics(glyphMetricsProperty.FindPropertyRelative("m_Width").floatValue, glyphMetricsProperty.FindPropertyRelative("m_Height").floatValue, glyphMetricsProperty.FindPropertyRelative("m_HorizontalBearingX").floatValue, glyphMetricsProperty.FindPropertyRelative("m_HorizontalBearingY").floatValue, glyphMetricsProperty.FindPropertyRelative("m_HorizontalAdvance").floatValue);

            character.glyph.scale = glyphProperty.FindPropertyRelative("m_Scale").floatValue;

            character.glyph.atlasIndex = glyphProperty.FindPropertyRelative("m_AtlasIndex").intValue;
        }


        void UpdateGlyphData(SerializedProperty property, int index)
        {
            Glyph glyph = m_fontAsset.glyphTable[index];

            glyph.index = (uint)property.FindPropertyRelative("m_Index").intValue;

            SerializedProperty glyphRect = property.FindPropertyRelative("m_GlyphRect");
            glyph.glyphRect = new GlyphRect(glyphRect.FindPropertyRelative("m_X").intValue, glyphRect.FindPropertyRelative("m_Y").intValue, glyphRect.FindPropertyRelative("m_Width").intValue, glyphRect.FindPropertyRelative("m_Height").intValue);

            SerializedProperty glyphMetrics = property.FindPropertyRelative("m_Metrics");
            glyph.metrics = new GlyphMetrics(glyphMetrics.FindPropertyRelative("m_Width").floatValue, glyphMetrics.FindPropertyRelative("m_Height").floatValue, glyphMetrics.FindPropertyRelative("m_HorizontalBearingX").floatValue, glyphMetrics.FindPropertyRelative("m_HorizontalBearingY").floatValue, glyphMetrics.FindPropertyRelative("m_HorizontalAdvance").floatValue);

            glyph.scale = property.FindPropertyRelative("m_Scale").floatValue;
        }


        void DisplayPageNavigation(ref int currentPage, int arraySize, int itemsPerPage)
        {
            Rect pagePos = EditorGUILayout.GetControlRect(false, 20);
            pagePos.width /= 3;

            int shiftMultiplier = Event.current.shift ? 10 : 1; // Page + Shift goes 10 page forward

            // Previous Page
            GUI.enabled = currentPage > 0;

            if (GUI.Button(pagePos, "Previous Page"))
                currentPage -= 1 * shiftMultiplier;


            // Page Counter
            GUI.enabled = true;
            pagePos.x += pagePos.width;
            int totalPages = (int)(arraySize / (float)itemsPerPage + 0.999f);
            GUI.Label(pagePos, "Page " + (currentPage + 1) + " / " + totalPages, TMP_UIStyleManager.centeredLabel);

            // Next Page
            pagePos.x += pagePos.width;
            GUI.enabled = itemsPerPage * (currentPage + 1) < arraySize;

            if (GUI.Button(pagePos, "Next Page"))
                currentPage += 1 * shiftMultiplier;

            // Clamp page range
            currentPage = Mathf.Clamp(currentPage, 0, arraySize / itemsPerPage);

            GUI.enabled = true;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="srcGlyphID"></param>
        /// <param name="dstGlyphID"></param>
        bool AddNewGlyph(int srcIndex, int dstGlyphID)
        {
            // Make sure Destination Glyph ID doesn't already contain a Glyph
            if (m_fontAsset.glyphLookupTable.ContainsKey((uint)dstGlyphID))
                return false;

            // Add new element to glyph list.
            m_GlyphTable_prop.arraySize += 1;

            // Get a reference to the source glyph.
            SerializedProperty sourceGlyph = m_GlyphTable_prop.GetArrayElementAtIndex(srcIndex);

            int dstIndex = m_GlyphTable_prop.arraySize - 1;

            // Get a reference to the target / destination glyph.
            SerializedProperty targetGlyph = m_GlyphTable_prop.GetArrayElementAtIndex(dstIndex);

            CopyGlyphSerializedProperty(sourceGlyph, ref targetGlyph);

            // Update the ID of the glyph
            targetGlyph.FindPropertyRelative("m_Index").intValue = dstGlyphID;

            serializedObject.ApplyModifiedProperties();

            m_fontAsset.SortGlyphTable();

            m_fontAsset.ReadFontAssetDefinition();

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="glyphID"></param>
        void RemoveGlyphFromList(int index)
        {
            if (index > m_GlyphTable_prop.arraySize)
                return;

            int targetGlyphIndex = m_GlyphTable_prop.GetArrayElementAtIndex(index).FindPropertyRelative("m_Index").intValue;

            m_GlyphTable_prop.DeleteArrayElementAtIndex(index);

            // Remove all characters referencing this glyph.
            for (int i = 0; i < m_CharacterTable_prop.arraySize; i++)
            {
                int glyphIndex = m_CharacterTable_prop.GetArrayElementAtIndex(i).FindPropertyRelative("m_GlyphIndex").intValue;

                if (glyphIndex == targetGlyphIndex)
                {
                    // Remove character
                    m_CharacterTable_prop.DeleteArrayElementAtIndex(i);
                }
            }

            serializedObject.ApplyModifiedProperties();

            m_fontAsset.ReadFontAssetDefinition();
        }

        bool AddNewCharacter(int srcIndex, int dstGlyphID)
        {
            // Make sure Destination Glyph ID doesn't already contain a Glyph
            if (m_fontAsset.characterLookupTable.ContainsKey((uint)dstGlyphID))
                return false;

            // Add new element to glyph list.
            m_CharacterTable_prop.arraySize += 1;

            // Get a reference to the source glyph.
            SerializedProperty sourceCharacter = m_CharacterTable_prop.GetArrayElementAtIndex(srcIndex);

            int dstIndex = m_CharacterTable_prop.arraySize - 1;

            // Get a reference to the target / destination glyph.
            SerializedProperty targetCharacter = m_CharacterTable_prop.GetArrayElementAtIndex(dstIndex);

            CopyCharacterSerializedProperty(sourceCharacter, ref targetCharacter);

            // Update the ID of the glyph
            targetCharacter.FindPropertyRelative("m_Unicode").intValue = dstGlyphID;

            serializedObject.ApplyModifiedProperties();

            m_fontAsset.SortCharacterTable();

            m_fontAsset.ReadFontAssetDefinition();

            return true;
        }

        void RemoveCharacterFromList(int index)
        {
            if (index > m_CharacterTable_prop.arraySize)
                return;

            m_CharacterTable_prop.DeleteArrayElementAtIndex(index);

            serializedObject.ApplyModifiedProperties();

            m_fontAsset.ReadFontAssetDefinition();
        }


        // Check if any of the Style elements were clicked on.
        private bool DoSelectionCheck(Rect selectionArea)
        {
            Event currentEvent = Event.current;

            switch (currentEvent.type)
            {
                case EventType.MouseDown:
                    if (selectionArea.Contains(currentEvent.mousePosition) && currentEvent.button == 0)
                    {
                        currentEvent.Use();
                        return true;
                    }

                    break;
            }

            return false;
        }

        TMP_GlyphValueRecord GetValueRecord(SerializedProperty property)
        {
            TMP_GlyphValueRecord record = new TMP_GlyphValueRecord();
            record.xPlacement = property.FindPropertyRelative("m_XPlacement").floatValue;
            record.yPlacement = property.FindPropertyRelative("m_YPlacement").floatValue;
            record.xAdvance = property.FindPropertyRelative("m_XAdvance").floatValue;
            record.yAdvance = property.FindPropertyRelative("m_YAdvance").floatValue;

            return record;
        }

        void RemoveAdjustmentPairFromList(int index)
        {
            if (index > m_GlyphPairAdjustmentRecords_prop.arraySize)
                return;

            m_GlyphPairAdjustmentRecords_prop.DeleteArrayElementAtIndex(index);

            serializedObject.ApplyModifiedProperties();

            m_fontAsset.ReadFontAssetDefinition();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="srcGlyph"></param>
        /// <param name="dstGlyph"></param>
        void CopyGlyphSerializedProperty(SerializedProperty srcGlyph, ref SerializedProperty dstGlyph)
        {
            // TODO : Should make a generic function which copies each of the properties.
            dstGlyph.FindPropertyRelative("m_Index").intValue = srcGlyph.FindPropertyRelative("m_Index").intValue;

            // Glyph -> GlyphMetrics
            SerializedProperty srcGlyphMetrics = srcGlyph.FindPropertyRelative("m_Metrics");
            SerializedProperty dstGlyphMetrics = dstGlyph.FindPropertyRelative("m_Metrics");

            dstGlyphMetrics.FindPropertyRelative("m_Width").floatValue = srcGlyphMetrics.FindPropertyRelative("m_Width").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_Height").floatValue = srcGlyphMetrics.FindPropertyRelative("m_Height").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalBearingX").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalBearingX").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalBearingY").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalBearingY").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalAdvance").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalAdvance").floatValue;

            // Glyph -> GlyphRect
            SerializedProperty srcGlyphRect = srcGlyph.FindPropertyRelative("m_GlyphRect");
            SerializedProperty dstGlyphRect = dstGlyph.FindPropertyRelative("m_GlyphRect");

            dstGlyphRect.FindPropertyRelative("m_X").intValue = srcGlyphRect.FindPropertyRelative("m_X").intValue;
            dstGlyphRect.FindPropertyRelative("m_Y").intValue = srcGlyphRect.FindPropertyRelative("m_Y").intValue;
            dstGlyphRect.FindPropertyRelative("m_Width").intValue = srcGlyphRect.FindPropertyRelative("m_Width").intValue;
            dstGlyphRect.FindPropertyRelative("m_Height").intValue = srcGlyphRect.FindPropertyRelative("m_Height").intValue;

            dstGlyph.FindPropertyRelative("m_Scale").floatValue = srcGlyph.FindPropertyRelative("m_Scale").floatValue;
            dstGlyph.FindPropertyRelative("m_AtlasIndex").intValue = srcGlyph.FindPropertyRelative("m_AtlasIndex").intValue;
        }


        void CopyCharacterSerializedProperty(SerializedProperty source, ref SerializedProperty target)
        {
            // TODO : Should make a generic function which copies each of the properties.
            int unicode = source.FindPropertyRelative("m_Unicode").intValue;
            target.FindPropertyRelative("m_Unicode").intValue = unicode;

            int srcGlyphIndex = source.FindPropertyRelative("m_GlyphIndex").intValue;
            target.FindPropertyRelative("m_GlyphIndex").intValue = srcGlyphIndex;

            target.FindPropertyRelative("m_Scale").floatValue = source.FindPropertyRelative("m_Scale").floatValue;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="searchPattern"></param>
        /// <returns></returns>
        void SearchGlyphTable (string searchPattern, ref List<int> searchResults)
        {
            if (searchResults == null) searchResults = new List<int>();

            searchResults.Clear();

            int arraySize = m_GlyphTable_prop.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                SerializedProperty sourceGlyph = m_GlyphTable_prop.GetArrayElementAtIndex(i);

                int id = sourceGlyph.FindPropertyRelative("m_Index").intValue;

                // Check for potential match against a character.
                //if (searchPattern.Length == 1 && id == searchPattern[0])
                //    searchResults.Add(i);

                // Check for potential match against decimal id
                if (id.ToString().Contains(searchPattern))
                    searchResults.Add(i);

                //if (id.ToString("x").Contains(searchPattern))
                //    searchResults.Add(i);

                //if (id.ToString("X").Contains(searchPattern))
                //    searchResults.Add(i);
            }
        }


        void SearchCharacterTable(string searchPattern, ref List<int> searchResults)
        {
            if (searchResults == null) searchResults = new List<int>();

            searchResults.Clear();

            int arraySize = m_CharacterTable_prop.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                SerializedProperty sourceCharacter = m_CharacterTable_prop.GetArrayElementAtIndex(i);

                int id = sourceCharacter.FindPropertyRelative("m_Unicode").intValue;

                // Check for potential match against a character.
                if (searchPattern.Length == 1 && id == searchPattern[0])
                    searchResults.Add(i);
                else if (id.ToString("x").Contains(searchPattern))
                    searchResults.Add(i);
                else if (id.ToString("X").Contains(searchPattern))
                    searchResults.Add(i);
                
                // Check for potential match against decimal id
                //if (id.ToString().Contains(searchPattern))
                //    searchResults.Add(i);
            }
        }


        void SearchKerningTable(string searchPattern, ref List<int> searchResults)
        {
            if (searchResults == null) searchResults = new List<int>();

            searchResults.Clear();

            // Lookup glyph index of potential characters contained in the search pattern.
            uint firstGlyphIndex = 0;
            TMP_Character firstCharacterSearch;

            if (searchPattern.Length > 0 && m_fontAsset.characterLookupTable.TryGetValue(searchPattern[0], out firstCharacterSearch))
                firstGlyphIndex = firstCharacterSearch.glyphIndex;

            uint secondGlyphIndex = 0;
            TMP_Character secondCharacterSearch;

            if (searchPattern.Length > 1 && m_fontAsset.characterLookupTable.TryGetValue(searchPattern[1], out secondCharacterSearch))
                secondGlyphIndex = secondCharacterSearch.glyphIndex;

            int arraySize = m_GlyphPairAdjustmentRecords_prop.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                SerializedProperty record = m_GlyphPairAdjustmentRecords_prop.GetArrayElementAtIndex(i);

                SerializedProperty firstAdjustmentRecord = record.FindPropertyRelative("m_FirstAdjustmentRecord");
                SerializedProperty secondAdjustmentRecord = record.FindPropertyRelative("m_SecondAdjustmentRecord");

                int firstGlyph = firstAdjustmentRecord.FindPropertyRelative("m_GlyphIndex").intValue;
                int secondGlyph = secondAdjustmentRecord.FindPropertyRelative("m_GlyphIndex").intValue;

                if (firstGlyphIndex == firstGlyph && secondGlyphIndex == secondGlyph)
                    searchResults.Add(i);
                else if (searchPattern.Length == 1 && (firstGlyphIndex == firstGlyph || firstGlyphIndex == secondGlyph))
                    searchResults.Add(i);
                else if (firstGlyph.ToString().Contains(searchPattern))
                    searchResults.Add(i);
                else if (secondGlyph.ToString().Contains(searchPattern))
                    searchResults.Add(i);
            }
        }
    }
}