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
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_SensorName)), true);

                EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_CellScale)), true);
                // We only supports 2D GridSensor now so display gridNumSide as Vector2
                var gridSize = so.FindProperty(nameof(GridSensorComponent.m_GridSize));
                var gridSize2d = new Vector2Int(gridSize.vector3IntValue.x, gridSize.vector3IntValue.z);
                var newGridSize = EditorGUILayout.Vector2IntField("Grid Size", gridSize2d);
                gridSize.vector3IntValue = new Vector3Int(newGridSize.x, 1, newGridSize.y);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_RotateWithAgent)), true);

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                // detectable tags
                var detectableTags = so.FindProperty(nameof(GridSensorComponent.m_DetectableTags));
                var newSize = EditorGUILayout.IntField("Detectable Tags", detectableTags.arraySize);
                if (newSize != detectableTags.arraySize)
                {
                    detectableTags.arraySize = newSize;
                }
                EditorGUI.indentLevel++;
                for (var i = 0; i < detectableTags.arraySize; i++)
                {
                    var objectTag = detectableTags.GetArrayElementAtIndex(i);
                    EditorGUILayout.PropertyField(objectTag, new GUIContent("Tag " + i), true);
                }
                EditorGUI.indentLevel--;

                EditorGUILayout.LabelField("Observation Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_UseOneHotTag)), new GUIContent("One-Hot Tag Index"), true);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_CountColliders)), new GUIContent("Detectable Tag Count"), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ColliderMask)), true);
            EditorGUILayout.LabelField("Sensor Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ObservationStacks)), true);
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_CompressionType)), true);
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.LabelField("Collider and Buffer", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_InitialColliderBufferSize)), true);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_MaxColliderBufferSize)), true);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.LabelField("Debug Gizmo", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ShowGizmos)), true);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_GizmoYOffset)), true);

            // detectable objects
            var debugColors = so.FindProperty(nameof(GridSensorComponent.m_DebugColors));
            var detectableObjectSize = so.FindProperty(nameof(GridSensorComponent.m_DetectableTags)).arraySize;
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
