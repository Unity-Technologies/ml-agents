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
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_RootReference)), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_RotateWithAgent)), true);

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.LabelField("Channel Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_DepthType)), true);

                // channel depth
                var channelDepth = so.FindProperty(nameof(GridSensorComponent.m_ChannelDepths));
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
                var detectableObjects = so.FindProperty(nameof(GridSensorComponent.m_DetectableObjects));
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
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_MaxColliderBufferSize)), true);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_InitialColliderBufferSize)), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ColliderMask)), true);
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.LabelField("Sensor Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ObservationStacks)), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_CompressionType)), true);

            EditorGUILayout.LabelField("Debug Gizmo", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_ShowGizmos)), true);
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_GizmoYOffset)), true);

            // detectable objects
            var debugColors = so.FindProperty(nameof(GridSensorComponent.m_DebugColors));
            var detectableObjectSize = so.FindProperty(nameof(GridSensorComponent.m_DetectableObjects)).arraySize;
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
