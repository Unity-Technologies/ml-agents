using UnityEditor;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Editor
{
    [CustomEditor(typeof(BufferSensorComponent), editorForChildClasses: true)]
    [CanEditMultipleObjects]
    internal class BufferSensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the BufferSensorComponent

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // These fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_ObservableSize"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_MaxNumObservables"), true);
            }
            EditorGUI.EndDisabledGroup();

            so.ApplyModifiedProperties();
        }
    }
}
