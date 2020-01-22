using UnityEngine;
using UnityEditor;

namespace TMPro.EditorUtilities
{

    [CustomEditor(typeof(TextMeshPro), true), CanEditMultipleObjects]
    public class TMP_EditorPanel : TMP_BaseEditorPanel
    {
        static readonly GUIContent k_SortingLayerLabel = new GUIContent("Sorting Layer", "Name of the Renderer's sorting layer.");
        static readonly GUIContent k_OrderInLayerLabel = new GUIContent("Order in Layer", "Renderer's order within a sorting layer.");
        static readonly GUIContent k_OrthographicLabel = new GUIContent("Orthographic Mode", "Should be enabled when using an orthographic camera. Instructs the shader to not perform any perspective correction.");
        static readonly GUIContent k_VolumetricLabel = new GUIContent("Volumetric Setup", "Use cubes rather than quads to render the text. Allows for volumetric rendering when combined with a compatible shader.");
        
        SerializedProperty m_IsVolumetricTextProp;
        
        SerializedProperty m_IsOrthographicProp;

        Renderer m_Renderer;

        protected override void OnEnable()
        {
            base.OnEnable();

            m_IsOrthographicProp = serializedObject.FindProperty("m_isOrthographic");
            
            m_IsVolumetricTextProp = serializedObject.FindProperty("m_isVolumetricText");

            m_Renderer = m_TextComponent.GetComponent<Renderer>();
        }

        protected override void DrawExtraSettings()
        {
            Foldout.extraSettings = EditorGUILayout.Foldout(Foldout.extraSettings, k_ExtraSettingsLabel, true, TMP_UIStyleManager.boldFoldout);
            if (Foldout.extraSettings)
            {
                EditorGUI.indentLevel += 1;

                DrawMargins();

                DrawSortingLayer();

                DrawGeometrySorting();

                DrawOrthographicMode();
                
                DrawRichText();

                DrawParsing();

                DrawVolumetricSetup();

                DrawKerning();

                DrawPadding();

                EditorGUI.indentLevel -= 1;
            }
        }

        protected void DrawSortingLayer()
        {
            Undo.RecordObject(m_Renderer, "Sorting Layer Change");

            EditorGUI.BeginChangeCheck();

            // SORTING LAYERS
            var sortingLayerNames = SortingLayerHelper.sortingLayerNames;

            var textComponent = (TextMeshPro)m_TextComponent;

            // Look up the layer name using the current layer ID
            string oldName = SortingLayerHelper.GetSortingLayerNameFromID(textComponent.sortingLayerID);

            // Use the name to look up our array index into the names list
            int oldLayerIndex = System.Array.IndexOf(sortingLayerNames, oldName);

            // Show the pop-up for the names
            EditorGUIUtility.fieldWidth = 0f;
            int newLayerIndex = EditorGUILayout.Popup(k_SortingLayerLabel, oldLayerIndex, sortingLayerNames);
            
            // If the index changes, look up the ID for the new index to store as the new ID
            if (newLayerIndex != oldLayerIndex)
            {
                textComponent.sortingLayerID = SortingLayerHelper.GetSortingLayerIDForIndex(newLayerIndex);
            }

            // Expose the manual sorting order
            int newSortingLayerOrder = EditorGUILayout.IntField(k_OrderInLayerLabel, textComponent.sortingOrder);
            if (newSortingLayerOrder != textComponent.sortingOrder)
            {
                textComponent.sortingOrder = newSortingLayerOrder;
            }

            if (EditorGUI.EndChangeCheck())
                m_HavePropertiesChanged = true;

            EditorGUILayout.Space();
        }

        protected void DrawOrthographicMode()
        {
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_IsOrthographicProp, k_OrthographicLabel);
            if (EditorGUI.EndChangeCheck())
                m_HavePropertiesChanged = true;
        }

        protected void DrawVolumetricSetup()
        {
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_IsVolumetricTextProp, k_VolumetricLabel);
            if (EditorGUI.EndChangeCheck())
            {
                m_HavePropertiesChanged = true;
                m_TextComponent.textInfo.ResetVertexLayout(m_IsVolumetricTextProp.boolValue);
            }

            EditorGUILayout.Space();
        }

        // Method to handle multi object selection
        protected override bool IsMixSelectionTypes()
        {
            GameObject[] objects = Selection.gameObjects;
            if (objects.Length > 1)
            {
                for (int i = 0; i < objects.Length; i++)
                {
                    if (objects[i].GetComponent<TextMeshPro>() == null)
                        return true;
                }
            }
            return false;
        }

        protected override void OnUndoRedo()
        {
            int undoEventId = Undo.GetCurrentGroup();
            int lastUndoEventId = s_EventId;

            if (undoEventId != lastUndoEventId)
            {
                for (int i = 0; i < targets.Length; i++)
                {
                    //Debug.Log("Undo & Redo Performed detected in Editor Panel. Event ID:" + Undo.GetCurrentGroup());
                    TMPro_EventManager.ON_TEXTMESHPRO_PROPERTY_CHANGED(true, targets[i] as TextMeshPro);
                    s_EventId = undoEventId;
                }
            }
        }
    }
}