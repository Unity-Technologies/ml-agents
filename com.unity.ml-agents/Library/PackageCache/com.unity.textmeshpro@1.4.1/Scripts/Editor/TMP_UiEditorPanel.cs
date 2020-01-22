using UnityEngine;
using UnityEngine.UI;
using UnityEditor;

namespace TMPro.EditorUtilities
{

    [CustomEditor(typeof(TextMeshProUGUI), true), CanEditMultipleObjects]
    public class TMP_UiEditorPanel : TMP_BaseEditorPanel
    {
        static readonly GUIContent k_RaycastTargetLabel = new GUIContent("Raycast Target", "Whether the text blocks raycasts from the Graphic Raycaster.");

        SerializedProperty m_RaycastTargetProp;

        protected override void OnEnable()
        {
            base.OnEnable();
            m_RaycastTargetProp = serializedObject.FindProperty("m_RaycastTarget");
        }

        protected override void DrawExtraSettings()
        {
            Foldout.extraSettings = EditorGUILayout.Foldout(Foldout.extraSettings, k_ExtraSettingsLabel, true, TMP_UIStyleManager.boldFoldout);
            if (Foldout.extraSettings)
            {
                EditorGUI.indentLevel += 1;

                DrawMargins();

                DrawGeometrySorting();

                DrawRichText();

                DrawRaycastTarget();

                DrawParsing();
                
                DrawKerning();

                DrawPadding();

                EditorGUI.indentLevel -= 1;
            }
        }

        protected void DrawRaycastTarget()
        {
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(m_RaycastTargetProp, k_RaycastTargetLabel);
            if (EditorGUI.EndChangeCheck())
            {
                // Change needs to propagate to the child sub objects.
                Graphic[] graphicComponents = m_TextComponent.GetComponentsInChildren<Graphic>();
                for (int i = 1; i < graphicComponents.Length; i++)
                    graphicComponents[i].raycastTarget = m_RaycastTargetProp.boolValue;

                m_HavePropertiesChanged = true;
            }
        }

        // Method to handle multi object selection
        protected override bool IsMixSelectionTypes()
        {
            GameObject[] objects = Selection.gameObjects;
            if (objects.Length > 1)
            {
                for (int i = 0; i < objects.Length; i++)
                {
					if (objects[i].GetComponent<TextMeshProUGUI>() == null)
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
                    TMPro_EventManager.ON_TEXTMESHPRO_UGUI_PROPERTY_CHANGED(true, targets[i] as TextMeshProUGUI);
                    s_EventId = undoEventId;
                }
            }
        }
    }
}