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
        bool ShowHierarchy = true;

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            var rbSensorComp = so.targetObject as RigidBodySensorComponent;


            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // These fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(so.FindProperty("RootBody"), true);
                EditorGUILayout.PropertyField(so.FindProperty("VirtualRoot"), true);
                EditorGUILayout.PropertyField(so.FindProperty("Settings"), true);
                var requireExtractorUpdate = EditorGUI.EndChangeCheck();
                so.ApplyModifiedProperties();
                if (requireExtractorUpdate)
                {
                    rbSensorComp.ResetPoseExtractor();
                }

                ShowHierarchy = EditorGUILayout.Foldout(ShowHierarchy, "Hierarchy", true);
                if (ShowHierarchy)
                {
                    var treeNodes = rbSensorComp.GetTreeNodes();
                    var originalIndent = EditorGUI.indentLevel;
                    foreach (var node in treeNodes)
                    {
                        var obj = node.NodeObject;
                        var objContents = EditorGUIUtility.ObjectContent(obj, obj.GetType());
                        EditorGUI.indentLevel = originalIndent + node.Depth;
                        var enabled = EditorGUILayout.Toggle(objContents, node.Enabled);
                        rbSensorComp.SetPoseEnabled(node.OriginalIndex, enabled);
                    }

                    EditorGUI.indentLevel = originalIndent;
                }


                EditorGUILayout.PropertyField(so.FindProperty("sensorName"), true);
            }
            EditorGUI.EndDisabledGroup();



        }


    }
}
