using UnityEngine;
using UnityEngine.TextCore;
using UnityEditor;
using UnityEditorInternal;
using System.Collections.Generic;




namespace TMPro.EditorUtilities
{

    [CustomEditor(typeof(TMP_SpriteAsset))]
    public class TMP_SpriteAssetEditor : Editor
    {
        struct UI_PanelState
        {
            public static bool spriteAssetInfoPanel = true;
            public static bool fallbackSpriteAssetPanel = true;
            public static bool spriteCharacterTablePanel;
            public static bool spriteGlyphTablePanel;
        }

        private static string[] s_UiStateLabel = new string[] { "<i>(Click to collapse)</i> ", "<i>(Click to expand)</i> " };

        int m_moveToIndex;
        int m_selectedElement = -1;
        int m_CurrentCharacterPage;
        int m_CurrentGlyphPage;

        const string k_UndoRedo = "UndoRedoPerformed";

        string m_CharacterSearchPattern;
        List<int> m_CharacterSearchList;
        bool m_IsCharacterSearchDirty;

        string m_GlyphSearchPattern;
        List<int> m_GlyphSearchList;
        bool m_IsGlyphSearchDirty;

        SerializedProperty m_spriteAtlas_prop;
        SerializedProperty m_material_prop;
        SerializedProperty m_SpriteCharacterTableProperty;
        SerializedProperty m_SpriteGlyphTableProperty;
        ReorderableList m_fallbackSpriteAssetList;

        TMP_SpriteAsset m_SpriteAsset;

        bool isAssetDirty;

        float m_xOffset;
        float m_yOffset;
        float m_xAdvance;
        float m_scale;

        public void OnEnable()
        {
            m_SpriteAsset = target as TMP_SpriteAsset;

            m_spriteAtlas_prop = serializedObject.FindProperty("spriteSheet");
            m_material_prop = serializedObject.FindProperty("material");
            m_SpriteCharacterTableProperty = serializedObject.FindProperty("m_SpriteCharacterTable");
            m_SpriteGlyphTableProperty = serializedObject.FindProperty("m_SpriteGlyphTable");

            // Fallback TMP Sprite Asset list
            m_fallbackSpriteAssetList = new ReorderableList(serializedObject, serializedObject.FindProperty("fallbackSpriteAssets"), true, true, true, true);

            m_fallbackSpriteAssetList.drawElementCallback = (Rect rect, int index, bool isActive, bool isFocused) =>
            {
                var element = m_fallbackSpriteAssetList.serializedProperty.GetArrayElementAtIndex(index);
                rect.y += 2;
                EditorGUI.PropertyField(new Rect(rect.x, rect.y, rect.width, EditorGUIUtility.singleLineHeight), element, GUIContent.none);
            };

            m_fallbackSpriteAssetList.drawHeaderCallback = rect =>
            {
                EditorGUI.LabelField(rect, new GUIContent("Fallback Sprite Asset List", "Select the Sprite Assets that will be searched and used as fallback when a given sprite is missing from this sprite asset."));
            };
        }


        public override void OnInspectorGUI()
        {

            //Debug.Log("OnInspectorGUI Called.");
            Event currentEvent = Event.current;
            string evt_cmd = currentEvent.commandName; // Get Current Event CommandName to check for Undo Events

            serializedObject.Update();

            Rect rect;

            // TEXTMESHPRO SPRITE INFO PANEL
            GUILayout.Label("Sprite Info", EditorStyles.boldLabel);
            EditorGUI.indentLevel = 1;
            
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_spriteAtlas_prop , new GUIContent("Sprite Atlas"));
            if (EditorGUI.EndChangeCheck())
            {
                // Assign the new sprite atlas texture to the current material
                Texture2D tex = m_spriteAtlas_prop.objectReferenceValue as Texture2D;
                if (tex != null)
                {
                    Material mat = m_material_prop.objectReferenceValue as Material;
                    if (mat != null)
                        mat.mainTexture = tex;
                }
            }

            EditorGUILayout.PropertyField(m_material_prop, new GUIContent("Default Material"));
             
            EditorGUILayout.Space();

            // FALLBACK SPRITE ASSETS
            EditorGUI.indentLevel = 0;
            UI_PanelState.fallbackSpriteAssetPanel = EditorGUILayout.Foldout(UI_PanelState.fallbackSpriteAssetPanel, new GUIContent("Fallback Sprite Assets", "Select the Sprite Assets that will be searched and used as fallback when a given sprite is missing from this sprite asset."), true, TMP_UIStyleManager.boldFoldout);
            
            if (UI_PanelState.fallbackSpriteAssetPanel)
            {
                m_fallbackSpriteAssetList.DoLayoutList();
            }

            // SPRITE CHARACTER TABLE
            #region Display Sprite Character Table
            EditorGUI.indentLevel = 0;
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Sprite Character Table</b>", "List of sprite characters contained in this sprite asset."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.spriteCharacterTablePanel = !UI_PanelState.spriteCharacterTablePanel;

            GUI.Label(rect, (UI_PanelState.spriteCharacterTablePanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.spriteCharacterTablePanel)
            {
                int arraySize = m_SpriteCharacterTableProperty.arraySize;
                int itemsPerPage = 10;

                // Display Glyph Management Tools
                EditorGUILayout.BeginVertical(EditorStyles.helpBox, GUILayout.ExpandWidth(true));
                {
                    // Search Bar implementation
                    #region DISPLAY SEARCH BAR
                    EditorGUILayout.BeginHorizontal();
                    {
                        EditorGUIUtility.labelWidth = 110f;
                        EditorGUI.BeginChangeCheck();
                        string searchPattern = EditorGUILayout.TextField("Sprite Search", m_CharacterSearchPattern, "SearchTextField");
                        if (EditorGUI.EndChangeCheck() || m_IsCharacterSearchDirty)
                        {
                            if (string.IsNullOrEmpty(searchPattern) == false)
                            {
                                //GUIUtility.keyboardControl = 0;
                                m_CharacterSearchPattern = searchPattern.ToLower(System.Globalization.CultureInfo.InvariantCulture).Trim();

                                // Search Glyph Table for potential matches
                                SearchCharacterTable(m_CharacterSearchPattern, ref m_CharacterSearchList);
                            }
                            else
                                m_CharacterSearchPattern = null;

                            m_IsCharacterSearchDirty = false;
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

                    // Display Page Navigation
                    DisplayPageNavigation(ref m_CurrentCharacterPage, arraySize, itemsPerPage);
                }
                EditorGUILayout.EndVertical();

                if (arraySize > 0)
                {
                    // Display each SpriteInfo entry using the SpriteInfo property drawer.
                    for (int i = itemsPerPage * m_CurrentCharacterPage; i < arraySize && i < itemsPerPage * (m_CurrentCharacterPage + 1); i++)
                    {
                        // Define the start of the selection region of the element.
                        Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        int elementIndex = i;
                        if (!string.IsNullOrEmpty(m_CharacterSearchPattern))
                            elementIndex = m_CharacterSearchList[i];

                        SerializedProperty spriteCharacterProperty = m_SpriteCharacterTableProperty.GetArrayElementAtIndex(elementIndex);

                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                        {
                            EditorGUI.BeginDisabledGroup(i != m_selectedElement);
                            {
                                EditorGUILayout.PropertyField(spriteCharacterProperty);
                            }
                            EditorGUI.EndDisabledGroup();
                        }
                        EditorGUILayout.EndVertical();

                        // Define the end of the selection region of the element.
                        Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        // Check for Item selection
                        Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                        if (DoSelectionCheck(selectionArea))
                        {
                            if (m_selectedElement == i)
                            {
                                m_selectedElement = -1;
                            }
                            else
                            {
                                m_selectedElement = i;
                                GUIUtility.keyboardControl = 0;
                            }
                        }

                        // Draw & Handle Section Area
                        if (m_selectedElement == i)
                        {
                            // Draw selection highlight
                            TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));

                            // Draw options to MoveUp, MoveDown, Add or Remove Sprites
                            Rect controlRect = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight * 1f);
                            controlRect.width /= 8;

                            // Move sprite up.
                            bool guiEnabled = GUI.enabled;
                            if (i == 0) { GUI.enabled = false; }
                            if (GUI.Button(controlRect, "Up"))
                            {
                                SwapCharacterElements(i, i - 1);
                            }
                            GUI.enabled = guiEnabled;

                            // Move sprite down.
                            controlRect.x += controlRect.width;
                            if (i == arraySize - 1) { GUI.enabled = false; }
                            if (GUI.Button(controlRect, "Down"))
                            {
                                SwapCharacterElements(i, i + 1);
                            }
                            GUI.enabled = guiEnabled;

                            // Move sprite to new index
                            controlRect.x += controlRect.width * 2;
                            //if (i == arraySize - 1) { GUI.enabled = false; }
                            m_moveToIndex = EditorGUI.IntField(controlRect, m_moveToIndex);
                            controlRect.x -= controlRect.width;
                            if (GUI.Button(controlRect, "Goto"))
                            {
                                MoveCharacterToIndex(i, m_moveToIndex);
                            }
                            //controlRect.x += controlRect.width;
                            GUI.enabled = guiEnabled;

                            // Add new Sprite
                            controlRect.x += controlRect.width * 4;
                            if (GUI.Button(controlRect, "+"))
                            {
                                m_SpriteCharacterTableProperty.arraySize += 1;

                                int index = m_SpriteCharacterTableProperty.arraySize - 1;

                                SerializedProperty spriteInfo_prop = m_SpriteCharacterTableProperty.GetArrayElementAtIndex(index);

                                // Copy properties of the selected element
                                CopyCharacterSerializedProperty(m_SpriteCharacterTableProperty.GetArrayElementAtIndex(elementIndex), ref spriteInfo_prop);

                                //spriteInfo_prop.FindPropertyRelative("m_Index").intValue = index;
                                serializedObject.ApplyModifiedProperties();

                                m_IsCharacterSearchDirty = true;
                            }

                            // Delete selected Sprite
                            controlRect.x += controlRect.width;
                            if (m_selectedElement == -1) GUI.enabled = false;
                            if (GUI.Button(controlRect, "-"))
                            {
                                m_SpriteCharacterTableProperty.DeleteArrayElementAtIndex(elementIndex);

                                m_selectedElement = -1;
                                serializedObject.ApplyModifiedProperties();

                                m_IsCharacterSearchDirty = true;

                                return;
                            }


                        }
                    }
                }

                DisplayPageNavigation(ref m_CurrentCharacterPage, arraySize, itemsPerPage);

                EditorGUIUtility.labelWidth = 40f;
                EditorGUIUtility.fieldWidth = 20f;

                GUILayout.Space(5f);

                // GLOBAL TOOLS
                #region Global Tools
                /*
                GUI.enabled = true;
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                rect = EditorGUILayout.GetControlRect(false, 40);
               
                float width = (rect.width - 75f) / 4;
                EditorGUI.LabelField(rect, "Global Offsets & Scale", EditorStyles.boldLabel);
                
                
                rect.x += 70;
                bool old_ChangedState = GUI.changed;

                GUI.changed = false;
                m_xOffset = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 0, rect.y + 20, width - 5f, 18), new GUIContent("OX:"), m_xOffset);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalBearingX", m_xOffset);
                
                m_yOffset = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 1, rect.y + 20, width - 5f, 18), new GUIContent("OY:"), m_yOffset);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalBearingY", m_yOffset);
                
                m_xAdvance = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 2, rect.y + 20, width - 5f, 18), new GUIContent("ADV."), m_xAdvance);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalAdvance", m_xAdvance);
                
                m_scale = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 3, rect.y + 20, width - 5f, 18), new GUIContent("SF."), m_scale);
                if (GUI.changed) UpdateGlobalProperty("m_Scale", m_scale);

                EditorGUILayout.EndVertical();
                
                GUI.changed = old_ChangedState;
                */
                #endregion
 
            }
            #endregion


            // SPRITE GLYPH TABLE
            #region Display Sprite Glyph Table
            EditorGUI.indentLevel = 0;
            rect = EditorGUILayout.GetControlRect(false, 24);

            if (GUI.Button(rect, new GUIContent("<b>Sprite Glyph Table</b>", "A list of the SpriteGlyphs contained in this sprite asset."), TMP_UIStyleManager.sectionHeader))
                UI_PanelState.spriteGlyphTablePanel = !UI_PanelState.spriteGlyphTablePanel;

            GUI.Label(rect, (UI_PanelState.spriteGlyphTablePanel ? "" : s_UiStateLabel[1]), TMP_UIStyleManager.rightLabel);

            if (UI_PanelState.spriteGlyphTablePanel)
            {
                int arraySize = m_SpriteGlyphTableProperty.arraySize;
                int itemsPerPage = 10;

                // Display Glyph Management Tools
                EditorGUILayout.BeginVertical(EditorStyles.helpBox, GUILayout.ExpandWidth(true));
                {
                    // Search Bar implementation
                    #region DISPLAY SEARCH BAR
                    EditorGUILayout.BeginHorizontal();
                    {
                        EditorGUIUtility.labelWidth = 110f;
                        EditorGUI.BeginChangeCheck();
                        string searchPattern = EditorGUILayout.TextField("Sprite Search", m_GlyphSearchPattern, "SearchTextField");
                        if (EditorGUI.EndChangeCheck() || m_IsGlyphSearchDirty)
                        {
                            if (string.IsNullOrEmpty(searchPattern) == false)
                            {
                                //GUIUtility.keyboardControl = 0;
                                m_GlyphSearchPattern = searchPattern.ToLower(System.Globalization.CultureInfo.InvariantCulture).Trim();

                                // Search Glyph Table for potential matches
                                SearchCharacterTable(m_GlyphSearchPattern, ref m_GlyphSearchList);
                            }
                            else
                                m_GlyphSearchPattern = null;

                            m_IsGlyphSearchDirty = false;
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

                    // Display Page Navigation
                    DisplayPageNavigation(ref m_CurrentGlyphPage, arraySize, itemsPerPage);
                }
                EditorGUILayout.EndVertical();

                if (arraySize > 0)
                {
                    // Display each SpriteInfo entry using the SpriteInfo property drawer.
                    for (int i = itemsPerPage * m_CurrentGlyphPage; i < arraySize && i < itemsPerPage * (m_CurrentGlyphPage + 1); i++)
                    {
                        // Define the start of the selection region of the element.
                        Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        int elementIndex = i;
                        if (!string.IsNullOrEmpty(m_GlyphSearchPattern))
                            elementIndex = m_GlyphSearchList[i];

                        SerializedProperty spriteGlyphProperty = m_SpriteGlyphTableProperty.GetArrayElementAtIndex(elementIndex);

                        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                        {
                            EditorGUI.BeginDisabledGroup(i != m_selectedElement);
                            {
                                EditorGUILayout.PropertyField(spriteGlyphProperty);
                            }
                            EditorGUI.EndDisabledGroup();
                        }
                        EditorGUILayout.EndVertical();

                        // Define the end of the selection region of the element.
                        Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                        // Check for Item selection
                        Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                        if (DoSelectionCheck(selectionArea))
                        {
                            if (m_selectedElement == i)
                            {
                                m_selectedElement = -1;
                            }
                            else
                            {
                                m_selectedElement = i;
                                GUIUtility.keyboardControl = 0;
                            }
                        }

                        // Draw & Handle Section Area
                        if (m_selectedElement == i)
                        {
                            // Draw selection highlight
                            TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));

                            // Draw options to MoveUp, MoveDown, Add or Remove Sprites
                            Rect controlRect = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight * 1f);
                            controlRect.width /= 8;

                            // Move sprite up.
                            bool guiEnabled = GUI.enabled;
                            if (i == 0) { GUI.enabled = false; }
                            if (GUI.Button(controlRect, "Up"))
                            {
                                SwapGlyphElements(i, i - 1);
                            }
                            GUI.enabled = guiEnabled;

                            // Move sprite down.
                            controlRect.x += controlRect.width;
                            if (i == arraySize - 1) { GUI.enabled = false; }
                            if (GUI.Button(controlRect, "Down"))
                            {
                                SwapGlyphElements(i, i + 1);
                            }
                            GUI.enabled = guiEnabled;

                            // Move sprite to new index
                            controlRect.x += controlRect.width * 2;
                            //if (i == arraySize - 1) { GUI.enabled = false; }
                            m_moveToIndex = EditorGUI.IntField(controlRect, m_moveToIndex);
                            controlRect.x -= controlRect.width;
                            if (GUI.Button(controlRect, "Goto"))
                            {
                                MoveGlyphToIndex(i, m_moveToIndex);
                            }
                            //controlRect.x += controlRect.width;
                            GUI.enabled = guiEnabled;

                            // Add new Sprite
                            controlRect.x += controlRect.width * 4;
                            if (GUI.Button(controlRect, "+"))
                            {
                                m_SpriteGlyphTableProperty.arraySize += 1;

                                int index = m_SpriteGlyphTableProperty.arraySize - 1;

                                SerializedProperty newSpriteGlyphProperty = m_SpriteGlyphTableProperty.GetArrayElementAtIndex(index);

                                // Copy properties of the selected element
                                CopyGlyphSerializedProperty(m_SpriteGlyphTableProperty.GetArrayElementAtIndex(elementIndex), ref newSpriteGlyphProperty);

                                newSpriteGlyphProperty.FindPropertyRelative("m_Index").intValue = index;

                                serializedObject.ApplyModifiedProperties();

                                m_IsGlyphSearchDirty = true;
                                
                                //m_SpriteAsset.UpdateLookupTables();
                            }

                            // Delete selected Sprite
                            controlRect.x += controlRect.width;
                            if (m_selectedElement == -1) GUI.enabled = false;
                            if (GUI.Button(controlRect, "-"))
                            {
                                SerializedProperty selectedSpriteGlyphProperty = m_SpriteGlyphTableProperty.GetArrayElementAtIndex(elementIndex);

                                int selectedGlyphIndex = selectedSpriteGlyphProperty.FindPropertyRelative("m_Index").intValue;

                                m_SpriteGlyphTableProperty.DeleteArrayElementAtIndex(elementIndex);

                                // Remove all Sprite Characters referencing this glyph.
                                for (int j = 0; j < m_SpriteCharacterTableProperty.arraySize; j++)
                                {
                                    int glyphIndex = m_SpriteCharacterTableProperty.GetArrayElementAtIndex(j).FindPropertyRelative("m_GlyphIndex").intValue;

                                    if (glyphIndex == selectedGlyphIndex)
                                    {
                                        // Remove character
                                        m_SpriteCharacterTableProperty.DeleteArrayElementAtIndex(j);
                                    }
                                }

                                m_selectedElement = -1;
                                serializedObject.ApplyModifiedProperties();

                                m_IsGlyphSearchDirty = true;

                                //m_SpriteAsset.UpdateLookupTables();

                                return;
                            }


                        }
                    }
                }

                DisplayPageNavigation(ref m_CurrentGlyphPage, arraySize, itemsPerPage);

                EditorGUIUtility.labelWidth = 40f;
                EditorGUIUtility.fieldWidth = 20f;

                GUILayout.Space(5f);

                // GLOBAL TOOLS
                #region Global Tools
                GUI.enabled = true;
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                rect = EditorGUILayout.GetControlRect(false, 40);

                float width = (rect.width - 75f) / 4;
                EditorGUI.LabelField(rect, "Global Offsets & Scale", EditorStyles.boldLabel);


                rect.x += 70;
                bool old_ChangedState = GUI.changed;

                GUI.changed = false;
                m_xOffset = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 0, rect.y + 20, width - 5f, 18), new GUIContent("OX:"), m_xOffset);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalBearingX", m_xOffset);

                m_yOffset = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 1, rect.y + 20, width - 5f, 18), new GUIContent("OY:"), m_yOffset);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalBearingY", m_yOffset);

                m_xAdvance = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 2, rect.y + 20, width - 5f, 18), new GUIContent("ADV."), m_xAdvance);
                if (GUI.changed) UpdateGlobalProperty("m_HorizontalAdvance", m_xAdvance);

                m_scale = EditorGUI.FloatField(new Rect(rect.x + 5f + width * 3, rect.y + 20, width - 5f, 18), new GUIContent("SF."), m_scale);
                if (GUI.changed) UpdateGlobalProperty("m_Scale", m_scale);

                EditorGUILayout.EndVertical();
                #endregion

                GUI.changed = old_ChangedState;

            }
            #endregion


            if (serializedObject.ApplyModifiedProperties() || evt_cmd == k_UndoRedo || isAssetDirty)
            {
                if (m_SpriteAsset.m_IsSpriteAssetLookupTablesDirty || evt_cmd == k_UndoRedo)
                    m_SpriteAsset.UpdateLookupTables();

                TMPro_EventManager.ON_SPRITE_ASSET_PROPERTY_CHANGED(true, m_SpriteAsset);

                isAssetDirty = false;
                EditorUtility.SetDirty(target);
            }

            // Clear selection if mouse event was not consumed. 
            GUI.enabled = true;
            if (currentEvent.type == EventType.MouseDown && currentEvent.button == 0)
                m_selectedElement = -1;

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="arraySize"></param>
        /// <param name="itemsPerPage"></param>
        void DisplayPageNavigation(ref int currentPage, int arraySize, int itemsPerPage)
        {
            Rect pagePos = EditorGUILayout.GetControlRect(false, 20);
            pagePos.width /= 3;

            int shiftMultiplier = Event.current.shift ? 10 : 1; // Page + Shift goes 10 page forward

            // Previous Page
            GUI.enabled = currentPage > 0;

            if (GUI.Button(pagePos, "Previous Page"))
            {
                currentPage -= 1 * shiftMultiplier;
                //m_isNewPage = true;
            }

            // Page Counter
            GUI.enabled = true;
            pagePos.x += pagePos.width;
            int totalPages = (int)(arraySize / (float)itemsPerPage + 0.999f);
            GUI.Label(pagePos, "Page " + (currentPage + 1) + " / " + totalPages, TMP_UIStyleManager.centeredLabel);

            // Next Page
            pagePos.x += pagePos.width;
            GUI.enabled = itemsPerPage * (currentPage + 1) < arraySize;

            if (GUI.Button(pagePos, "Next Page"))
            {
                currentPage += 1 * shiftMultiplier;
                //m_isNewPage = true;
            }

            // Clamp page range
            currentPage = Mathf.Clamp(currentPage, 0, arraySize / itemsPerPage);

            GUI.enabled = true;
        }


        /// <summary>
        /// Method to update the properties of all sprites
        /// </summary>
        /// <param name="property"></param>
        /// <param name="value"></param>
        void UpdateGlobalProperty(string property, float value)
        {
            int arraySize = m_SpriteGlyphTableProperty.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                // Get a reference to the sprite glyph.
                SerializedProperty spriteGlyphProperty = m_SpriteGlyphTableProperty.GetArrayElementAtIndex(i);

                if (property == "m_Scale")
                {
                    spriteGlyphProperty.FindPropertyRelative(property).floatValue = value;
                }
                else
                {
                    SerializedProperty glyphMetricsProperty = spriteGlyphProperty.FindPropertyRelative("m_Metrics");
                    glyphMetricsProperty.FindPropertyRelative(property).floatValue = value;
                }
            }

            GUI.changed = false;
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


        /// <summary>
        /// Swap the sprite item at the currently selected array index to another index.
        /// </summary>
        /// <param name="selectedIndex">Selected index.</param>
        /// <param name="newIndex">New index.</param>
        void SwapCharacterElements(int selectedIndex, int newIndex)
        {
            m_SpriteCharacterTableProperty.MoveArrayElement(selectedIndex, newIndex);
            m_selectedElement = newIndex;
            m_IsCharacterSearchDirty = true;
            m_SpriteAsset.m_IsSpriteAssetLookupTablesDirty = true;
        }

        /// <summary>
        /// Move Sprite Element at selected index to another index and reorder sprite list.
        /// </summary>
        /// <param name="selectedIndex"></param>
        /// <param name="newIndex"></param>
        void MoveCharacterToIndex(int selectedIndex, int newIndex)
        {
            int arraySize = m_SpriteCharacterTableProperty.arraySize;

            if (newIndex >= arraySize)
                newIndex = arraySize - 1;

            m_SpriteCharacterTableProperty.MoveArrayElement(selectedIndex, newIndex);

            m_selectedElement = newIndex;
            m_IsCharacterSearchDirty = true;
            m_SpriteAsset.m_IsSpriteAssetLookupTablesDirty = true;

            // TODO: Need to handle switching pages if the character or glyph is moved to a different page.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="selectedIndex"></param>
        /// <param name="newIndex"></param>
        void SwapGlyphElements(int selectedIndex, int newIndex)
        {
            m_SpriteGlyphTableProperty.MoveArrayElement(selectedIndex, newIndex);
            m_selectedElement = newIndex;
            m_IsGlyphSearchDirty = true;
            m_SpriteAsset.m_IsSpriteAssetLookupTablesDirty = true;
        }

        /// <summary>
        /// Move Sprite Element at selected index to another index and reorder sprite list.
        /// </summary>
        /// <param name="selectedIndex"></param>
        /// <param name="newIndex"></param>
        void MoveGlyphToIndex(int selectedIndex, int newIndex)
        {
            int arraySize = m_SpriteGlyphTableProperty.arraySize;

            if (newIndex >= arraySize)
                newIndex = arraySize - 1;

            m_SpriteGlyphTableProperty.MoveArrayElement(selectedIndex, newIndex);

            m_selectedElement = newIndex;
            m_IsGlyphSearchDirty = true;
            m_SpriteAsset.m_IsSpriteAssetLookupTablesDirty = true;

            // TODO: Need to handle switching pages if the character or glyph is moved to a different page.
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        void CopyCharacterSerializedProperty(SerializedProperty source, ref SerializedProperty target)
        {
            target.FindPropertyRelative("m_Name").stringValue = source.FindPropertyRelative("m_Name").stringValue;
            target.FindPropertyRelative("m_HashCode").intValue = source.FindPropertyRelative("m_HashCode").intValue;
            target.FindPropertyRelative("m_Unicode").intValue = source.FindPropertyRelative("m_Unicode").intValue;
            target.FindPropertyRelative("m_GlyphIndex").intValue = source.FindPropertyRelative("m_GlyphIndex").intValue;
            target.FindPropertyRelative("m_Scale").floatValue = source.FindPropertyRelative("m_Scale").floatValue;
        }

        void CopyGlyphSerializedProperty(SerializedProperty srcGlyph, ref SerializedProperty dstGlyph)
        {
            // TODO : Should make a generic function which copies each of the properties.

            // Index
            dstGlyph.FindPropertyRelative("m_Index").intValue = srcGlyph.FindPropertyRelative("m_Index").intValue;

            // GlyphMetrics
            SerializedProperty srcGlyphMetrics = srcGlyph.FindPropertyRelative("m_Metrics");
            SerializedProperty dstGlyphMetrics = dstGlyph.FindPropertyRelative("m_Metrics");

            dstGlyphMetrics.FindPropertyRelative("m_Width").floatValue = srcGlyphMetrics.FindPropertyRelative("m_Width").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_Height").floatValue = srcGlyphMetrics.FindPropertyRelative("m_Height").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalBearingX").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalBearingX").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalBearingY").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalBearingY").floatValue;
            dstGlyphMetrics.FindPropertyRelative("m_HorizontalAdvance").floatValue = srcGlyphMetrics.FindPropertyRelative("m_HorizontalAdvance").floatValue;

            // GlyphRect
            SerializedProperty srcGlyphRect = srcGlyph.FindPropertyRelative("m_GlyphRect");
            SerializedProperty dstGlyphRect = dstGlyph.FindPropertyRelative("m_GlyphRect");

            dstGlyphRect.FindPropertyRelative("m_X").intValue = srcGlyphRect.FindPropertyRelative("m_X").intValue;
            dstGlyphRect.FindPropertyRelative("m_Y").intValue = srcGlyphRect.FindPropertyRelative("m_Y").intValue;
            dstGlyphRect.FindPropertyRelative("m_Width").intValue = srcGlyphRect.FindPropertyRelative("m_Width").intValue;
            dstGlyphRect.FindPropertyRelative("m_Height").intValue = srcGlyphRect.FindPropertyRelative("m_Height").intValue;

            dstGlyph.FindPropertyRelative("m_Scale").floatValue = srcGlyph.FindPropertyRelative("m_Scale").floatValue;
            dstGlyph.FindPropertyRelative("m_AtlasIndex").intValue = srcGlyph.FindPropertyRelative("m_AtlasIndex").intValue;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="searchPattern"></param>
        /// <returns></returns>
        void SearchCharacterTable(string searchPattern, ref List<int> searchResults)
        {
            if (searchResults == null) searchResults = new List<int>();
            searchResults.Clear();

            int arraySize = m_SpriteCharacterTableProperty.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                SerializedProperty sourceSprite = m_SpriteCharacterTableProperty.GetArrayElementAtIndex(i);

                // Check for potential match against array index
                if (i.ToString().Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }

                // Check for potential match against decimal id
                int id = sourceSprite.FindPropertyRelative("m_GlyphIndex").intValue;
                if (id.ToString().Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }

                // Check for potential match against name
                string name = sourceSprite.FindPropertyRelative("m_Name").stringValue.ToLower(System.Globalization.CultureInfo.InvariantCulture).Trim();
                if (name.Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }
            }
        }

        void SearchGlyphTable(string searchPattern, ref List<int> searchResults)
        {
            if (searchResults == null) searchResults = new List<int>();
            searchResults.Clear();

            int arraySize = m_SpriteGlyphTableProperty.arraySize;

            for (int i = 0; i < arraySize; i++)
            {
                SerializedProperty sourceSprite = m_SpriteGlyphTableProperty.GetArrayElementAtIndex(i);

                // Check for potential match against array index
                if (i.ToString().Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }

                // Check for potential match against decimal id
                int id = sourceSprite.FindPropertyRelative("m_GlyphIndex").intValue;
                if (id.ToString().Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }

                // Check for potential match against name
                string name = sourceSprite.FindPropertyRelative("m_Name").stringValue.ToLower(System.Globalization.CultureInfo.InvariantCulture).Trim();
                if (name.Contains(searchPattern))
                {
                    searchResults.Add(i);
                    continue;
                }
            }
        }

    }
}