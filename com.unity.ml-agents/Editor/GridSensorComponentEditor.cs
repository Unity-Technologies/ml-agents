using UnityEditor;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Editor
{
    [CustomEditor(typeof(GridSensorComponent), editorForChildClasses: true)]
    [CanEditMultipleObjects]
    internal class GridSensorComponentEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
#if !MLA_UNITY_PHYSICS_MODULE
            EditorGUILayout.HelpBox("The Physics Module is not currently present.  " +
                "Please add it to your project in order to use the GridSensor APIs in the " +
                $"{nameof(GridSensorComponent)}", MessageType.Warning);
#endif

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
                // We only supports 2D GridSensor now so lock gridSize.y to 1
                var gridSize = so.FindProperty(nameof(GridSensorComponent.m_GridSize));
                var gridSize2d = new Vector3Int(gridSize.vector3IntValue.x, 1, gridSize.vector3IntValue.z);
                var newGridSize = EditorGUILayout.Vector3IntField("Grid Size", gridSize2d);
                gridSize.vector3IntValue = new Vector3Int(newGridSize.x, 1, newGridSize.z);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(nameof(GridSensorComponent.m_AgentGameObject)), true);
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
            EditorGUILayout.LabelField("Debug Colors");
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
