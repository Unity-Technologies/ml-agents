 using UnityEngine;
using UnityEngine.TextCore;
using UnityEngine.TextCore.LowLevel;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(Glyph))]
    public class TMP_GlyphPropertyDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            SerializedProperty prop_GlyphIndex = property.FindPropertyRelative("m_Index");
            SerializedProperty prop_GlyphMetrics = property.FindPropertyRelative("m_Metrics");
            SerializedProperty prop_GlyphRect = property.FindPropertyRelative("m_GlyphRect");
            SerializedProperty prop_Scale = property.FindPropertyRelative("m_Scale");
            SerializedProperty prop_AtlasIndex = property.FindPropertyRelative("m_AtlasIndex");

            GUIStyle style = new GUIStyle(EditorStyles.label);
            style.richText = true;

            Rect rect = new Rect(position.x + 70, position.y, position.width, 49);

            float labelWidth = GUI.skin.label.CalcSize(new GUIContent("ID: " + prop_GlyphIndex.intValue)).x;
            EditorGUI.LabelField(new Rect(position.x + (64 - labelWidth) / 2, position.y + 85, 64f, 18f), new GUIContent("ID: <color=#FFFF80>" + prop_GlyphIndex.intValue + "</color>"), style);
            //EditorGUIUtility.labelWidth = 22f;
            //EditorGUI.DelayedIntField(new Rect(position.x + (64 - labelWidth) / 2, position.y + 89, 58f, 18f), prop_GlyphIndex, new GUIContent("ID:"));

            // We get Rect since a valid position may not be provided by the caller.
            EditorGUI.PropertyField(new Rect(rect.x, rect.y, position.width, 49), prop_GlyphRect);

            rect.y += 45;
            EditorGUI.PropertyField(rect, prop_GlyphMetrics);

            EditorGUIUtility.labelWidth = 40f;
            EditorGUI.PropertyField(new Rect(rect.x, rect.y + 65, 75, 18), prop_Scale, new GUIContent("Scale:")); // new GUIContent("Scale: <color=#FFFF80>" + prop_Scale.floatValue + "</color>"), style);

            EditorGUIUtility.labelWidth = 74f;
            EditorGUI.PropertyField(new Rect(rect.x + 85, rect.y + 65, 95, 18), prop_AtlasIndex, new GUIContent("Atlas Index:")); //  new GUIContent("Atlas Index: <color=#FFFF80>" + prop_AtlasIndex.intValue + "</color>"), style);

            DrawGlyph(position, property);
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return 130f;
        }

        void DrawGlyph(Rect position, SerializedProperty property)
        {
            // Get a reference to the sprite texture
            TMP_FontAsset fontAsset = property.serializedObject.targetObject as TMP_FontAsset;

            if (fontAsset == null)
                return;

            // Get reference to atlas texture.
            int atlasIndex = property.FindPropertyRelative("m_AtlasIndex").intValue;
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

            // Draw glyph from atlas texture.
            Rect glyphDrawPosition = new Rect(position.x, position.y + 2, 64, 80);

            SerializedProperty prop_GlyphRect = property.FindPropertyRelative("m_GlyphRect");

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
