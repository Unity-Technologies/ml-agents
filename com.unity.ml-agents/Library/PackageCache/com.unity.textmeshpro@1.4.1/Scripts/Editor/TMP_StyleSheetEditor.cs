using UnityEngine;
using UnityEditor;


namespace TMPro.EditorUtilities
{

    [CustomPropertyDrawer(typeof(TMP_Style))]
    public class StyleDrawer : PropertyDrawer
    {
        public static readonly float height = 95f;

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return height;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            SerializedProperty nameProperty = property.FindPropertyRelative("m_Name");
            SerializedProperty hashCodeProperty = property.FindPropertyRelative("m_HashCode");
            SerializedProperty openingDefinitionProperty = property.FindPropertyRelative("m_OpeningDefinition");
            SerializedProperty closingDefinitionProperty = property.FindPropertyRelative("m_ClosingDefinition");
            SerializedProperty openingDefinitionArray = property.FindPropertyRelative("m_OpeningTagArray");
            SerializedProperty closingDefinitionArray = property.FindPropertyRelative("m_ClosingTagArray");


            EditorGUIUtility.labelWidth = 90;
            position.height = EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing;
            float labelHeight = position.height + 2f;

            EditorGUI.BeginChangeCheck();
            Rect rect0 = new Rect(position.x, position.y, (position.width) / 2 + 5, position.height);
            EditorGUI.PropertyField(rect0, nameProperty);
            if (EditorGUI.EndChangeCheck())
            {
                // Recompute HashCode if name has changed.
                hashCodeProperty.intValue = TMP_TextUtilities.GetSimpleHashCode(nameProperty.stringValue);

                property.serializedObject.ApplyModifiedProperties();
                // Dictionary needs to be updated since HashCode has changed.
                TMP_StyleSheet.RefreshStyles();
            }

            // HashCode
            Rect rect1 = new Rect(rect0.x + rect0.width + 5, position.y, 65, position.height);
            GUI.Label(rect1, "HashCode");
            GUI.enabled = false;
            rect1.x += 65;
            rect1.width = position.width / 2 - 75;
            EditorGUI.PropertyField(rect1, hashCodeProperty, GUIContent.none);
            
            GUI.enabled = true;

            // Text Tags
            EditorGUI.BeginChangeCheck();
            
            // Opening Tags
            position.y += labelHeight;
            GUI.Label(position, "Opening Tags");
            Rect textRect1 = new Rect(108, position.y, position.width - 86, 35);
            openingDefinitionProperty.stringValue = EditorGUI.TextArea(textRect1, openingDefinitionProperty.stringValue);
            if (EditorGUI.EndChangeCheck())
            {
                // If any properties have changed, we need to update the Opening and Closing Arrays.
                int size = openingDefinitionProperty.stringValue.Length;

                // Adjust array size to match new string length.
                if (openingDefinitionArray.arraySize != size) openingDefinitionArray.arraySize = size;

                for (int i = 0; i < size; i++)
                {
                    SerializedProperty element = openingDefinitionArray.GetArrayElementAtIndex(i);
                    element.intValue = openingDefinitionProperty.stringValue[i];
                }
            }

            EditorGUI.BeginChangeCheck();

            // Closing Tags
            position.y += 38;
            GUI.Label(position, "Closing Tags");
            Rect textRect2 = new Rect(108, position.y, position.width - 86, 35);
            closingDefinitionProperty.stringValue = EditorGUI.TextArea(textRect2, closingDefinitionProperty.stringValue);

            if (EditorGUI.EndChangeCheck())
            {
                // If any properties have changed, we need to update the Opening and Closing Arrays.
                int size = closingDefinitionProperty.stringValue.Length;
                
                // Adjust array size to match new string length.
                if (closingDefinitionArray.arraySize != size) closingDefinitionArray.arraySize = size;
                
                for (int i = 0; i < size; i++)
                {
                    SerializedProperty element = closingDefinitionArray.GetArrayElementAtIndex(i);
                    element.intValue = closingDefinitionProperty.stringValue[i];
                }            
            }

        }
    }



    [CustomEditor(typeof(TMP_StyleSheet)), CanEditMultipleObjects]
    public class TMP_StyleEditor : Editor
    {

        SerializedProperty m_StyleListProp;

        int m_SelectedElement = -1;

        //private Event m_CurrentEvent;
        int m_Page;


       
        void OnEnable()
        {
            m_StyleListProp = serializedObject.FindProperty("m_StyleList");
        }


        public override void OnInspectorGUI()
        {
            Event currentEvent = Event.current;

            serializedObject.Update();

            int arraySize = m_StyleListProp.arraySize;
            int itemsPerPage = (Screen.height - 178) / 111;

            if (arraySize > 0)
            {
                // Display each Style entry using the StyleDrawer PropertyDrawer.
                for (int i = itemsPerPage * m_Page; i < arraySize && i < itemsPerPage * (m_Page + 1); i++)
                {

                    // Define the start of the selection region of the element.
                    Rect elementStartRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                    EditorGUILayout.BeginVertical(EditorStyles.helpBox);
   
                    SerializedProperty spriteInfo = m_StyleListProp.GetArrayElementAtIndex(i);
                    EditorGUI.BeginChangeCheck();
                    EditorGUILayout.PropertyField(spriteInfo);
                    EditorGUILayout.EndVertical();
                    if (EditorGUI.EndChangeCheck())
                    {
                        //
                    }
      
                    // Define the end of the selection region of the element.
                    Rect elementEndRegion = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));

                    // Check for Item selection
                    Rect selectionArea = new Rect(elementStartRegion.x, elementStartRegion.y, elementEndRegion.width, elementEndRegion.y - elementStartRegion.y);
                    if (DoSelectionCheck(selectionArea))
                    {
                        if (m_SelectedElement == i)
                        {
                            m_SelectedElement = -1;
                        }
                        else
                        {
                            m_SelectedElement = i;
                            GUIUtility.keyboardControl = 0;
                        }
                    }
                    
                    // Handle Selection Highlighting
                    if (m_SelectedElement == i)
                    {
                        TMP_EditorUtility.DrawBox(selectionArea, 2f, new Color32(40, 192, 255, 255));
                    }
                }
            }

            int shiftMultiplier = currentEvent.shift ? 10 : 1; // Page + Shift goes 10 page forward

            GUILayout.Space(-3f);

            Rect pagePos = EditorGUILayout.GetControlRect(false, 20);
            pagePos.width /= 6;

            // Return if we can't display any items.
            if (itemsPerPage == 0) return;


            // Add new style.
            pagePos.x += pagePos.width * 4;
            if (GUI.Button(pagePos, "+"))
            {
                m_StyleListProp.arraySize += 1;
                serializedObject.ApplyModifiedProperties();
                TMP_StyleSheet.RefreshStyles();
            }


            // Delete selected style.
            pagePos.x += pagePos.width;
            if (m_SelectedElement == -1) GUI.enabled = false;
            if (GUI.Button(pagePos, "-"))
            {
                if (m_SelectedElement != -1)
                    m_StyleListProp.DeleteArrayElementAtIndex(m_SelectedElement);

                m_SelectedElement = -1;
                serializedObject.ApplyModifiedProperties();
                TMP_StyleSheet.RefreshStyles();
            }

            GUILayout.Space(5f);

            pagePos = EditorGUILayout.GetControlRect(false, 20);
            pagePos.width /= 3;

           
            // Previous Page
            if (m_Page > 0) GUI.enabled = true;
            else GUI.enabled = false;

            if (GUI.Button(pagePos, "Previous"))
                m_Page -= 1 * shiftMultiplier;

            // PAGE COUNTER
            GUI.enabled = true;
            pagePos.x += pagePos.width;
            int totalPages = (int)(arraySize / (float)itemsPerPage + 0.999f);
            GUI.Label(pagePos, "Page " + (m_Page + 1) + " / " + totalPages, TMP_UIStyleManager.centeredLabel);

            // Next Page
            pagePos.x += pagePos.width;
            if (itemsPerPage * (m_Page + 1) < arraySize) GUI.enabled = true;
            else GUI.enabled = false;

            if (GUI.Button(pagePos, "Next"))
                m_Page += 1 * shiftMultiplier;

            // Clamp page range
            m_Page = Mathf.Clamp(m_Page, 0, arraySize / itemsPerPage);


            if (serializedObject.ApplyModifiedProperties())
                TMPro_EventManager.ON_TEXT_STYLE_PROPERTY_CHANGED(true);

            // Clear selection if mouse event was not consumed. 
            GUI.enabled = true;
            if (currentEvent.type == EventType.MouseDown && currentEvent.button == 0)
                m_SelectedElement = -1;
            

        }


        // Check if any of the Style elements were clicked on.
        static bool DoSelectionCheck(Rect selectionArea)
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

    }

}
