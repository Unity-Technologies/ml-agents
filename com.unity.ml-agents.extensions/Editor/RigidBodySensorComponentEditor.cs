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

            var rbSensorComp = so.targetObject as RigidBodySensorComponent;

            // Drawing the CameraSensorComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // These fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUILayout.PropertyField(so.FindProperty("RootBody"), true);
                EditorGUILayout.PropertyField(so.FindProperty("VirtualRoot"), true);
                EditorGUILayout.PropertyField(so.FindProperty("Settings"), true);

                var treeNodes = rbSensorComp.GetTreeNodes();
                var originalIndent = EditorGUI.indentLevel;
                foreach (var node in treeNodes)
                {
                    var obj = node.NodeObject;
                    var objContents = EditorGUIUtility.ObjectContent(obj, obj.GetType());
                    EditorGUI.indentLevel = originalIndent + node.Depth;
                    EditorGUILayout.Toggle(objContents, node.Enabled);
                }

                EditorGUI.indentLevel = originalIndent;

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
