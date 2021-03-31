using UnityEditor;
using UnityEngine;
using Unity.MLAgents.Editor;
using Unity.MLAgents.Extensions.Sensors;

namespace Unity.MLAgents.Extensions.Editor
{
    [CustomEditor(typeof(GridSensorComponent))]
    [CanEditMultipleObjects]
    internal class GridSensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the GridSensorComponent
            EditorGUI.BeginChangeCheck();

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // These fields affect the sensor order or observation size,
                // So can't be changed at runtime.
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);

                EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty("m_CellScaleX"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_CellScaleZ"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_CellScaleY"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_GridNumSideX"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_GridNumSideZ"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_RotateWithAgent"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_RootReference"), true);

                EditorGUILayout.LabelField("Channel Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty("m_DepthType"), true);

                // channel depth
                var channelDepth = so.FindProperty("m_ChannelDepth");
                var newDepth = EditorGUILayout.IntField("Channel Depth", channelDepth.arraySize);
                if (newDepth != channelDepth.arraySize)
                {
                    channelDepth.arraySize = newDepth;
                }
                EditorGUI.indentLevel++;
                for (var i = 0; i < channelDepth.arraySize; i++)
                {
                    var objectTag = channelDepth.GetArrayElementAtIndex(i);
                    EditorGUILayout.PropertyField(objectTag, new GUIContent("Channel " + i + " Depth"), true);
                }
                EditorGUI.indentLevel--;

                // detectable objects
                var detectableObjects = so.FindProperty("m_DetectableObjects");
                var newSize = EditorGUILayout.IntField("Detectable Objects", detectableObjects.arraySize);
                if (newSize != detectableObjects.arraySize)
                {
                    detectableObjects.arraySize = newSize;
                }
                EditorGUI.indentLevel++;
                for (var i = 0; i < detectableObjects.arraySize; i++)
                {
                    var objectTag = detectableObjects.GetArrayElementAtIndex(i);
                    EditorGUILayout.PropertyField(objectTag, new GUIContent("Tag " + i), true);
                }
                EditorGUI.indentLevel--;

                EditorGUILayout.LabelField("Collider and Buffer", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty("m_MaxColliderBufferSize"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_InitialColliderBufferSize"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_ObserveMask"), true);

                EditorGUILayout.LabelField("Sensor Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty("m_ObservationStacks"), true);

            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty("m_CompressionType"), true);

            EditorGUILayout.LabelField("Debug Gizmo", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty("m_ShowGizmos"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_GizmoYOffset"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_DebugColors"), true);

            // detectable objects
            var debugColors = so.FindProperty("m_DebugColors");
            var detectableObjectSize = so.FindProperty("m_DetectableObjects").arraySize;
            if (detectableObjectSize != debugColors.arraySize)
            {
                debugColors.arraySize = detectableObjectSize;
            }
            EditorGUI.indentLevel++;
            for (var i = 0; i < debugColors.arraySize; i++)
            {
                var debugColor = debugColors.GetArrayElementAtIndex(i);
                EditorGUILayout.PropertyField(debugColor, new GUIContent("Tag " + i + " Color"), true);
            }
            EditorGUI.indentLevel--;

            var requireSensorUpdate = EditorGUI.EndChangeCheck();
            so.ApplyModifiedProperties();

            if (requireSensorUpdate)
            {
                UpdateSensor();
            }
        }

        void UpdateSensor()
        {
            var sensorComponent = serializedObject.targetObject as GridSensorComponent;
            sensorComponent?.UpdateSensor();
        }
    }
}
