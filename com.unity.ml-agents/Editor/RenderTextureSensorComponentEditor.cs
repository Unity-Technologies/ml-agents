using UnityEngine;
using UnityEditor;
using MLAgents.Sensors;
namespace MLAgents.Editor
{
    [CustomEditor(typeof(RenderTextureSensorComponent))]
    [CanEditMultipleObjects]
    internal class RenderTextureSensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the RenderTextureComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(Application.isPlaying);
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_RenderTexture"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
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
