using UnityEngine;
using UnityEditor;
using MLAgents.Sensors;

namespace MLAgents.Editor
{
    [CustomEditor(typeof(CameraSensorComponent))]
    [CanEditMultipleObjects]
    internal class CameraSensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the CameraSensorComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(Application.isPlaying);
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_Camera"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_Width"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_Height"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_Grayscale"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_Compression"), true);
            }

            EditorGUI.EndDisabledGroup();

            if (EditorGUI.EndChangeCheck())
            {
                //
            }

            so.ApplyModifiedProperties();
        }
    }
}
