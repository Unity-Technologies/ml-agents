using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Editor;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Editor
{
    [CustomEditor(typeof(RigidBodySensorComponent))]
    [CanEditMultipleObjects]
    internal class RigidBodySensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the CameraSensorComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // These fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUILayout.PropertyField(so.FindProperty("RootBody"), true);
                EditorGUILayout.PropertyField(so.FindProperty("VirtualRoot"), true);
                EditorGUILayout.PropertyField(so.FindProperty("Settings"), true);

                // Draw the tree of bodies
                EditorGUILayout.Toggle("torso", false);
                EditorGUI.indentLevel++;
                EditorGUILayout.Toggle("leftArm", true);
                EditorGUI.indentLevel++;
                EditorGUILayout.Toggle("leftForearm", true);
                EditorGUI.indentLevel--;
                EditorGUI.indentLevel--;

                EditorGUI.indentLevel++;
                EditorGUILayout.Toggle("rightArm", true);
                EditorGUI.indentLevel++;
                EditorGUILayout.Toggle("rightForearm", true);
                EditorGUI.indentLevel--;
                EditorGUI.indentLevel--;

                EditorGUILayout.PropertyField(so.FindProperty("sensorName"), true);
            }
            EditorGUI.EndDisabledGroup();

            var requireSensorUpdate = EditorGUI.EndChangeCheck();
            so.ApplyModifiedProperties();

            if (requireSensorUpdate)
            {

            }
        }


    }
}
