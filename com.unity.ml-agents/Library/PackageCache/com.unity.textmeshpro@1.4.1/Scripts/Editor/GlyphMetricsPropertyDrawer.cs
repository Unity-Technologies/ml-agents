using UnityEngine;
using UnityEngine.TextCore;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(GlyphMetrics))]
    public class GlyphMetricsPropertyDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            SerializedProperty prop_Width = property.FindPropertyRelative("m_Width");
            SerializedProperty prop_Height = property.FindPropertyRelative("m_Height");
            SerializedProperty prop_HoriBearingX = property.FindPropertyRelative("m_HorizontalBearingX");
            SerializedProperty prop_HoriBearingY = property.FindPropertyRelative("m_HorizontalBearingY");
            SerializedProperty prop_HoriAdvance = property.FindPropertyRelative("m_HorizontalAdvance");

            // We get Rect since a valid position may not be provided by the caller.
            Rect rect = new Rect(position.x, position.y, position.width, 49);

            EditorGUI.LabelField(rect, new GUIContent("Glyph Metrics"));

            EditorGUIUtility.labelWidth = 30f;
            EditorGUIUtility.fieldWidth = 10f;

            //GUI.enabled = false;
            float width = (rect.width - 75f) / 2;
            EditorGUI.PropertyField(new Rect(rect.x + width * 0, rect.y + 20, width - 5f, 18), prop_Width, new GUIContent("W:"));
            EditorGUI.PropertyField(new Rect(rect.x + width * 1, rect.y + 20, width - 5f, 18), prop_Height, new GUIContent("H:"));

            //GUI.enabled = true;

            width = (rect.width - 75f) / 3;
            EditorGUI.BeginChangeCheck();
            EditorGUI.PropertyField(new Rect(rect.x + width * 0, rect.y + 40, width - 5f, 18), prop_HoriBearingX, new GUIContent("BX:"));
            EditorGUI.PropertyField(new Rect(rect.x + width * 1, rect.y + 40, width - 5f, 18), prop_HoriBearingY, new GUIContent("BY:"));
            EditorGUI.PropertyField(new Rect(rect.x + width * 2, rect.y + 40, width - 5f, 18), prop_HoriAdvance, new GUIContent("AD:"));
            if (EditorGUI.EndChangeCheck())
            {

            }
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return 65f;
        }

    }
}
