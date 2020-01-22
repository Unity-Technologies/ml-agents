using UnityEngine;
using UnityEditor;
using System.Collections;


namespace TMPro.EditorUtilities
{
    [CustomEditor(typeof(TextContainer)), CanEditMultipleObjects]
    public class TMPro_TextContainerEditor : Editor
    {
        
        // Serialized Properties 
        private SerializedProperty anchorPosition_prop;
        private SerializedProperty pivot_prop;
        private SerializedProperty rectangle_prop;
        private SerializedProperty margins_prop;
       

        private TextContainer m_textContainer;
        //private Transform m_transform;
        //private Vector3[] m_Rect_handlePoints = new Vector3[4];
        //private Vector3[] m_Margin_handlePoints = new Vector3[4];

        //private Vector2 m_anchorPosition;

        //private Vector3 m_mousePreviousPOS;
        //private Vector2 m_previousStartPOS;
        //private int m_mouseDragFlag = 0;

        //private static Transform m_visualHelper;


        void OnEnable()
        {
         
            // Serialized Properties
            anchorPosition_prop = serializedObject.FindProperty("m_anchorPosition");
            pivot_prop = serializedObject.FindProperty("m_pivot");
            rectangle_prop = serializedObject.FindProperty("m_rect"); 
            margins_prop = serializedObject.FindProperty("m_margins");

            m_textContainer = (TextContainer)target;
            //m_transform = m_textContainer.transform;
            

            /*
            if (m_visualHelper == null)
            {
                m_visualHelper = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                m_visualHelper.localScale = new Vector3(0.25f, 0.25f, 0.25f);
            }
            */
        }

        void OnDisable()
        {
            /*
            if (m_visualHelper != null)
                DestroyImmediate (m_visualHelper.gameObject);
            */
        }

     


        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(anchorPosition_prop);
            if (anchorPosition_prop.enumValueIndex == 9)
            {
                EditorGUI.indentLevel += 1;
                EditorGUILayout.PropertyField(pivot_prop, new GUIContent("Pivot Position"));
                EditorGUI.indentLevel -= 1;
            }


            DrawDimensionProperty(rectangle_prop, "Dimensions");
            DrawMaginProperty(margins_prop, "Margins");
            if (EditorGUI.EndChangeCheck())
            {
                // Re-compute pivot position when changes are made.
                if (anchorPosition_prop.enumValueIndex != 9)
                    pivot_prop.vector2Value = GetAnchorPosition(anchorPosition_prop.enumValueIndex);
                
                m_textContainer.hasChanged = true;
            }

            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Space();
        }


        private void DrawDimensionProperty(SerializedProperty property, string label)
        {
            float old_LabelWidth = EditorGUIUtility.labelWidth;
            float old_FieldWidth = EditorGUIUtility.fieldWidth;
                     
            Rect rect = EditorGUILayout.GetControlRect(false, 18);
            Rect pos0 = new Rect(rect.x, rect.y + 2, rect.width, 18);

            float width = rect.width + 3;
            pos0.width = old_LabelWidth;
            GUI.Label(pos0, label);

            Rect rectangle = property.rectValue;
            
            float width_B = width - old_LabelWidth;
            float fieldWidth = width_B / 4;
            pos0.width = fieldWidth - 5;

            pos0.x = old_LabelWidth + 15;
            GUI.Label(pos0, "Width");

            pos0.x += fieldWidth; 
            rectangle.width = EditorGUI.FloatField(pos0, GUIContent.none, rectangle.width);

            pos0.x += fieldWidth;
            GUI.Label(pos0, "Height");

            pos0.x += fieldWidth; 
            rectangle.height = EditorGUI.FloatField(pos0, GUIContent.none, rectangle.height);

            property.rectValue = rectangle;
            EditorGUIUtility.labelWidth = old_LabelWidth;
            EditorGUIUtility.fieldWidth = old_FieldWidth;
        }


        private void DrawMaginProperty(SerializedProperty property, string label)
        {
            float old_LabelWidth = EditorGUIUtility.labelWidth;
            float old_FieldWidth = EditorGUIUtility.fieldWidth;
          
            Rect rect = EditorGUILayout.GetControlRect(false, 2 * 18);
            Rect pos0 = new Rect(rect.x, rect.y + 2, rect.width, 18);

            float width = rect.width + 3;
            pos0.width = old_LabelWidth;
            GUI.Label(pos0, label);

            //Vector4 vec = property.vector4Value;
            Vector4 vec = Vector4.zero;
            vec.x = property.FindPropertyRelative("x").floatValue;
            vec.y = property.FindPropertyRelative("y").floatValue;
            vec.z = property.FindPropertyRelative("z").floatValue;
            vec.w = property.FindPropertyRelative("w").floatValue;


            float widthB = width - old_LabelWidth;
            float fieldWidth = widthB / 4;
            pos0.width = fieldWidth - 5;

            // Labels
            pos0.x = old_LabelWidth + 15;
            GUI.Label(pos0, "Left");

            pos0.x += fieldWidth;
            GUI.Label(pos0, "Top");

            pos0.x += fieldWidth;
            GUI.Label(pos0, "Right");

            pos0.x += fieldWidth;
            GUI.Label(pos0, "Bottom");

            pos0.y += 18;

            pos0.x = old_LabelWidth + 15;
            vec.x = EditorGUI.FloatField(pos0, GUIContent.none, vec.x);

            pos0.x += fieldWidth;
            vec.y = EditorGUI.FloatField(pos0, GUIContent.none, vec.y);

            pos0.x += fieldWidth;
            vec.z = EditorGUI.FloatField(pos0, GUIContent.none, vec.z);

            pos0.x += fieldWidth;
            vec.w = EditorGUI.FloatField(pos0, GUIContent.none, vec.w);

            //property.vector4Value = vec;         
            property.FindPropertyRelative("x").floatValue = vec.x;
            property.FindPropertyRelative("y").floatValue = vec.y;
            property.FindPropertyRelative("z").floatValue = vec.z;
            property.FindPropertyRelative("w").floatValue = vec.w;

            EditorGUIUtility.labelWidth = old_LabelWidth;
            EditorGUIUtility.fieldWidth = old_FieldWidth;
        }


        Vector2 GetAnchorPosition(int index)
        {
            Vector2 anchorPosition = Vector2.zero;
                   
            switch (index)
            {
                case 0: // TOP LEFT
                    anchorPosition = new Vector2(0, 1);
                    break;
                case 1: // TOP
                    anchorPosition = new Vector2(0.5f, 1);
                    break;
                case 2: // TOP RIGHT
                    anchorPosition = new Vector2(1, 1);
                    break;
                case 3: // LEFT
                    anchorPosition = new Vector2(0, 0.5f);
                    break;
                case 4: // MIDDLE
                    anchorPosition = new Vector2(0.5f, 0.5f);
                    break;
                case 5: // RIGHT
                    anchorPosition = new Vector2(1, 0.5f);
                    break;
                case 6: // BOTTOM LEFT
                    anchorPosition = new Vector2(0, 0);
                    break;
                case 7: // BOTTOM
                    anchorPosition = new Vector2(0.5f, 0);
                    break;
                case 8: // BOTTOM RIGHT
                    anchorPosition = new Vector2(1, 0);
                    break;
            }
          
            return anchorPosition;
        }


    }
}
