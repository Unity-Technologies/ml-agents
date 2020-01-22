using UnityEngine;
using UnityEngine.TextCore;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(TMP_SpriteGlyph))]
    public class TMP_SpriteGlyphPropertyDrawer : PropertyDrawer
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

            // Draw GlyphRect
            EditorGUI.PropertyField(rect, prop_GlyphRect);

            // Draw GlyphMetrics
            rect.y += 45;
            EditorGUI.PropertyField(rect, prop_GlyphMetrics);

            EditorGUIUtility.labelWidth = 40f;
            EditorGUI.PropertyField(new Rect(rect.x, rect.y + 65, 75, 18), prop_Scale, new GUIContent("Scale:"));

            EditorGUIUtility.labelWidth = 74f;
            EditorGUI.PropertyField(new Rect(rect.x + 85, rect.y + 65, 95, 18), prop_AtlasIndex, new GUIContent("Atlas Index:"));

            DrawGlyph(position, property);

            int spriteCharacterIndex;

            int.TryParse(property.displayName.Split(' ')[1], out spriteCharacterIndex);
            EditorGUI.LabelField(new Rect(position.x, position.y + 5, 64f, 18f), new GUIContent("#" + spriteCharacterIndex), style);

            float labelWidthID = GUI.skin.label.CalcSize(new GUIContent("ID: " + prop_GlyphIndex.intValue)).x;
            EditorGUI.LabelField(new Rect(position.x + (64 - labelWidthID) / 2, position.y + 110, 64f, 18f), new GUIContent("ID: <color=#FFFF80>" + prop_GlyphIndex.intValue + "</color>"), style);
        }

        void DrawGlyph(Rect position, SerializedProperty property)
        {
            // Get a reference to the sprite texture
            Texture tex = (property.serializedObject.targetObject as TMP_SpriteAsset).spriteSheet;

            // Return if we don't have a texture assigned to the sprite asset.
            if (tex == null)
            {
                Debug.LogWarning("Please assign a valid Sprite Atlas texture to the [" + property.serializedObject.targetObject.name + "] Sprite Asset.", property.serializedObject.targetObject);
                return;
            }

            Vector2 spriteTexPosition = new Vector2(position.x, position.y);
            Vector2 spriteSize = new Vector2(65, 65);

            SerializedProperty prop_GlyphRect = property.FindPropertyRelative("m_GlyphRect");

            int spriteImageX = prop_GlyphRect.FindPropertyRelative("m_X").intValue;
            int spriteImageY = prop_GlyphRect.FindPropertyRelative("m_Y").intValue;
            int spriteImageWidth = prop_GlyphRect.FindPropertyRelative("m_Width").intValue;
            int spriteImageHeight = prop_GlyphRect.FindPropertyRelative("m_Height").intValue;

            if (spriteImageWidth >= spriteImageHeight)
            {
                spriteSize.y = spriteImageHeight * spriteSize.x / spriteImageWidth;
                spriteTexPosition.y += (spriteSize.x - spriteSize.y) / 2;
            }
            else
            {
                spriteSize.x = spriteImageWidth * spriteSize.y / spriteImageHeight;
                spriteTexPosition.x += (spriteSize.y - spriteSize.x) / 2;
            }

            // Compute the normalized texture coordinates
            Rect texCoords = new Rect((float)spriteImageX / tex.width, (float)spriteImageY / tex.height, (float)spriteImageWidth / tex.width, (float)spriteImageHeight / tex.height);
            GUI.DrawTextureWithTexCoords(new Rect(spriteTexPosition.x + 5, spriteTexPosition.y + 32f, spriteSize.x, spriteSize.y), tex, texCoords, true);
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return 130f;
        }

    }
}
