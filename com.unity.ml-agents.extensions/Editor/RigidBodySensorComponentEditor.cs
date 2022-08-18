using Unity.MLAgents.Editor;
using Unity.MLAgents.Extensions.Sensors;
using UnityEditor;

namespace Unity.MLAgents.Extensions.Editor
{
    [CustomEditor(typeof(RigidBodySensorComponent))]
    internal class RigidBodySensorComponentEditor : UnityEditor.Editor
    {
        bool ShowHierarchy = true;

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            var rbSensorComp = so.targetObject as RigidBodySensorComponent;
            if (rbSensorComp.IsTrivial())
            {
                EditorGUILayout.HelpBox(
                    "The Root Body has no Joints, and the Virtual Root is null or the same as the " +
                    "Root Body's GameObject. This will not generate any useful observations; they will always " +
                    "be the identity values. Consider removing this component since it won't help the Agent.",
                    MessageType.Warning
                );
            }

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // All the fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUI.BeginChangeCheck();
                EditorGUILayout.PropertyField(so.FindProperty("RootBody"), true);
                EditorGUILayout.PropertyField(so.FindProperty("VirtualRoot"), true);

                // Changing the root body or virtual root changes the hierarchy, so we need to reset.
                if (EditorGUI.EndChangeCheck())
                {
                    rbSensorComp.ResetPoseExtractor();
                }

                EditorGUILayout.PropertyField(so.FindProperty("Settings"), true);

                // Collapsible tree for the body hierarchy
                ShowHierarchy = EditorGUILayout.Foldout(ShowHierarchy, "Hierarchy", true);
                if (ShowHierarchy)
                {
                    var treeNodes = rbSensorComp.GetDisplayNodes();
                    var originalIndent = EditorGUI.indentLevel;
                    var poseEnabled = so.FindProperty("m_PoseExtractor").FindPropertyRelative("m_PoseEnabled");
                    foreach (var node in treeNodes)
                    {
                        var obj = node.NodeObject;
                        var objContents = EditorGUIUtility.ObjectContent(obj, obj.GetType());
                        EditorGUI.indentLevel = originalIndent + node.Depth;
                        EditorGUILayout.PropertyField(poseEnabled.GetArrayElementAtIndex(node.OriginalIndex), objContents);
                    }

                    EditorGUI.indentLevel = originalIndent;
                }

                EditorGUILayout.PropertyField(so.FindProperty("sensorName"), true);
            }
            EditorGUI.EndDisabledGroup();

            so.ApplyModifiedProperties();
        }
    }
}
