using UnityEngine;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{
    [CustomPropertyDrawer(typeof(TMP_Character))]
    public class TMP_CharacterPropertyDrawer : PropertyDrawer
    {
        //[SerializeField]
        //static Material s_InternalSDFMaterial;

        //[SerializeField]
        //static Material s_InternalBitmapMaterial;

        int m_GlyphSelectedForEditing = -1;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            SerializedProperty prop_Unicode = property.FindPropertyRelative("m_Unicode");
            SerializedProperty prop_GlyphIndex = property.FindPropertyRelative("m_GlyphIndex");
            SerializedProperty prop_Scale = property.FindPropertyRelative("m_Scale");


            GUIStyle style = new GUIStyle(EditorStyles.label);
            style.richText = true;

            EditorGUIUtility.labelWidth = 40f;
            EditorGUIUtility.fieldWidth = 50;

            Rect rect = new Rect(position.x + 50, position.y, position.width, 49);

            // Display non-editable fields
            if (GUI.enabled == false)
            {
                int unicode = prop_Unicode.intValue;
                EditorGUI.LabelField(new Rect(rect.x, rect.y, 120f, 18), new GUIContent("Unicode: <color=#FFFF80>0x" + unicode.ToString("X") + "</color>"), style);
                EditorGUI.LabelField(new Rect(rect.x + 115, rect.y, 120f, 18), unicode <= 0xFFFF ? new GUIContent("UTF16: <color=#FFFF80>\\u" + unicode.ToString("X4") + "</color>") : new GUIContent("UTF32: <color=#FFFF80>\\U" + unicode.ToString("X8") + "</color>"), style);
                EditorGUI.LabelField(new Rect(rect.x, rect.y + 18, 120, 18), new GUIContent("Glyph ID: <color=#FFFF80>" + prop_GlyphIndex.intValue + "</color>"), style);
                EditorGUI.LabelField(new Rect(rect.x, rect.y + 36, 80, 18), new GUIContent("Scale: <color=#FFFF80>" + prop_Scale.floatValue + "</color>"), style);

                // Draw Glyph (if exists)
                DrawGlyph(position, property);
            }
            else // Display editable fields
            {
                EditorGUIUtility.labelWidth = 55f;
                GUI.SetNextControlName("Unicode Input");
                EditorGUI.BeginChangeCheck();
                string unicode = EditorGUI.TextField(new Rect(rect.x, rect.y, 120, 18), "Unicode:", prop_Unicode.intValue.ToString("X"));

                if (GUI.GetNameOfFocusedControl() == "Unicode Input")
                {
                    //Filter out unwanted characters.
                    char chr = Event.current.character;
                    if ((chr < '0' || chr > '9') && (chr < 'a' || chr > 'f') && (chr < 'A' || chr > 'F'))
                    {
                        Event.current.character = '\0';
                    }
                }

                if (EditorGUI.EndChangeCheck())
                {
                    // Update Unicode value
                    prop_Unicode.intValue = TMP_TextUtilities.StringHexToInt(unicode);
                }

                // Cache current glyph index in case it needs to be restored if the new glyph index is invalid.
                int currentGlyphIndex = prop_GlyphIndex.intValue;

                EditorGUIUtility.labelWidth = 59f;
                EditorGUI.BeginChangeCheck();
                EditorGUI.DelayedIntField(new Rect(rect.x, rect.y + 18, 100, 18), prop_GlyphIndex, new GUIContent("Glyph ID:"));
                if (EditorGUI.EndChangeCheck())
                {
                    // Get a reference to the font asset
                    TMP_FontAsset fontAsset = property.serializedObject.targetObject as TMP_FontAsset;
                    
                    // Make sure new glyph index is valid.
                    int elementIndex = fontAsset.glyphTable.FindIndex(item => item.index == prop_GlyphIndex.intValue);

                    if (elementIndex == -1)
                        prop_GlyphIndex.intValue = currentGlyphIndex;
                    else
                        fontAsset.m_IsFontAssetLookupTablesDirty = true;
                }

                int glyphIndex = prop_GlyphIndex.intValue;
                
                // Reset glyph selection if new character has been selected.
                if (GUI.enabled && m_GlyphSelectedForEditing != glyphIndex)
                    m_GlyphSelectedForEditing = -1;

                // Display button to edit the glyph data.
                if (GUI.Button(new Rect(rect.x + 120, rect.y + 18, 75, 18), new GUIContent("Edit Glyph")))
                {
                    if (m_GlyphSelectedForEditing == -1)
                        m_GlyphSelectedForEditing = glyphIndex;
                    else
                        m_GlyphSelectedForEditing = -1;

                    // Button clicks should not result in potential change.
                    GUI.changed = false;
                }

                // Show the glyph property drawer if selected
                if (glyphIndex == m_GlyphSelectedForEditing && GUI.enabled)
                {
                    // Get a reference to the font asset
                    TMP_FontAsset fontAsset = property.serializedObject.targetObject as TMP_FontAsset;

                    if (fontAsset != null)
                    {
                        // Get the index of the glyph in the font asset glyph table.
                        int elementIndex = fontAsset.glyphTable.FindIndex(item => item.index == glyphIndex);
                        
                        if (elementIndex != -1)
                        {
                            SerializedProperty prop_GlyphTable = property.serializedObject.FindProperty("m_GlyphTable");
                            SerializedProperty prop_Glyph = prop_GlyphTable.GetArrayElementAtIndex(elementIndex);

                            SerializedProperty prop_GlyphMetrics = prop_Glyph.FindPropertyRelative("m_Metrics");
                            SerializedProperty prop_GlyphRect = prop_Glyph.FindPropertyRelative("m_GlyphRect");

                            Rect newRect = EditorGUILayout.GetControlRect(false, 115);
                            EditorGUI.DrawRect(new Rect(newRect.x + 52, newRect.y - 20, newRect.width - 52, newRect.height - 5), new Color(0.1f, 0.1f, 0.1f, 0.45f));
                            EditorGUI.DrawRect(new Rect(newRect.x + 53, newRect.y - 19, newRect.width - 54, newRect.height - 7), new Color(0.3f, 0.3f, 0.3f, 0.8f));

                            // Display GlyphRect
                            newRect.x += 55;
                            newRect.y -= 18;
                            newRect.width += 5;
                            EditorGUI.PropertyField(newRect, prop_GlyphRect);

                            // Display GlyphMetrics
                            newRect.y += 45;
                            EditorGUI.PropertyField(newRect, prop_GlyphMetrics);

                            rect.y += 120;
                        }
                    }
                }

                EditorGUIUtility.labelWidth = 39f;
                EditorGUI.PropertyField(new Rect(rect.x, rect.y + 36, 80, 18), prop_Scale, new GUIContent("Scale:"));
                
                // Draw Glyph (if exists)
                DrawGlyph(position, property);
            }
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return 58;
        }

        void DrawGlyph(Rect position, SerializedProperty property)
        {
            // Get a reference to the atlas texture
            TMP_FontAsset fontAsset = property.serializedObject.targetObject as TMP_FontAsset;

            if (fontAsset == null)
                return;

            // Get a reference to the Glyph Table
            SerializedProperty prop_GlyphTable = property.serializedObject.FindProperty("m_GlyphTable");
            int glyphIndex = property.FindPropertyRelative("m_GlyphIndex").intValue;
            int elementIndex = fontAsset.glyphTable.FindIndex(item => item.index == glyphIndex);

            // Return if we can't find the glyph
            if (elementIndex == -1)
                return;

            SerializedProperty prop_Glyph = prop_GlyphTable.GetArrayElementAtIndex(elementIndex);

            // Get reference to atlas texture.
            int atlasIndex = prop_Glyph.FindPropertyRelative("m_AtlasIndex").intValue;
            Texture2D atlasTexture = fontAsset.atlasTextures.Length > atlasIndex ? fontAsset.atlasTextures[atlasIndex] : null;

            if (atlasTexture == null)
                return;

            Material mat;
            if (((GlyphRasterModes)fontAsset.atlasRenderMode & GlyphRasterModes.RASTER_MODE_BITMAP) == GlyphRasterModes.RASTER_MODE_BITMAP)
            {
                mat = TMP_FontAssetEditor.internalBitmapMaterial;

                if (mat == null)
                    return;

                mat.mainTexture = atlasTexture;
                mat.SetColor("_Color", Color.white);
            }
            else
            {
                mat = TMP_FontAssetEditor.internalSDFMaterial;

                if (mat == null)
                    return;

                mat.mainTexture = atlasTexture;
                mat.SetFloat(ShaderUtilities.ID_GradientScale, fontAsset.atlasPadding + 1);
            }

            // Draw glyph
            Rect glyphDrawPosition = new Rect(position.x, position.y, 48, 58);

            SerializedProperty prop_GlyphRect = prop_Glyph.FindPropertyRelative("m_GlyphRect");

            int glyphOriginX = prop_GlyphRect.FindPropertyRelative("m_X").intValue;
            int glyphOriginY = prop_GlyphRect.FindPropertyRelative("m_Y").intValue;
            int glyphWidth = prop_GlyphRect.FindPropertyRelative("m_Width").intValue;
            int glyphHeight = prop_GlyphRect.FindPropertyRelative("m_Height").intValue;

            float normalizedHeight = fontAsset.faceInfo.ascentLine - fontAsset.faceInfo.descentLine;
            float scale = glyphDrawPosition.width / normalizedHeight;

            // Compute the normalized texture coordinates
            Rect texCoords = new Rect((float)glyphOriginX / atlasTexture.width, (float)glyphOriginY / atlasTexture.height, (float)glyphWidth / atlasTexture.width, (float)glyphHeight / atlasTexture.height);

            if (Event.current.type == EventType.Repaint)
            {
                glyphDrawPosition.x += (glyphDrawPosition.width - glyphWidth * scale) / 2;
                glyphDrawPosition.y += (glyphDrawPosition.height - glyphHeight * scale) / 2;
                glyphDrawPosition.width = glyphWidth * scale;
                glyphDrawPosition.height = glyphHeight * scale;

                // Could switch to using the default material of the font asset which would require passing scale to the shader.
                Graphics.DrawTexture(glyphDrawPosition, atlasTexture, texCoords, 0, 0, 0, 0, new Color(1f, 1f, 1f), mat);
            }
        }

    }
}
