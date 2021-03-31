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
        const string k_SensorName = "m_SensorName";
        const string k_CellScaleXName = "m_CellScaleX";
        const string k_CellScaleZName = "m_CellScaleZ";
        const string k_CellScaleYName = "m_CellScaleY";
        const string k_GridNumSideXName = "m_GridNumSideX";
        const string k_GridNumSideZName = "m_GridNumSideZ";
        const string k_RotateWithAgentName = "m_RotateWithAgent";
        const string k_RootReferenceName = "m_RootReference";
        const string k_DepthTypeName = "m_DepthType";
        const string k_ChannelDepthName = "m_ChannelDepth";
        const string k_DetectableObjectsName = "m_DetectableObjects";
        const string k_MaxColliderBufferSizeName = "m_MaxColliderBufferSize";
        const string k_InitialColliderBufferSizeName = "m_InitialColliderBufferSize";
        const string k_ObserveMaskName = "m_ObserveMask";
        const string k_ObservationStacksName = "m_ObservationStacks";
        const string k_CompressionTypeName = "m_CompressionType";
        const string k_ShowGizmosName = "m_ShowGizmos";
        const string k_GizmoYOffsetName = "m_GizmoYOffset";
        const string k_DebugColorsName = "m_DebugColors";

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
                EditorGUILayout.PropertyField(so.FindProperty(k_SensorName), true);

                EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(k_CellScaleXName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_CellScaleZName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_CellScaleYName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_GridNumSideXName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_GridNumSideZName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_RootReferenceName), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(k_RotateWithAgentName), true);

            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.LabelField("Channel Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(k_DepthTypeName), true);

                // channel depth
                var channelDepth = so.FindProperty(k_ChannelDepthName);
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
                var detectableObjects = so.FindProperty(k_DetectableObjectsName);
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
                EditorGUILayout.PropertyField(so.FindProperty(k_MaxColliderBufferSizeName), true);
                EditorGUILayout.PropertyField(so.FindProperty(k_InitialColliderBufferSizeName), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(k_ObserveMaskName), true);
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.LabelField("Sensor Settings", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(so.FindProperty(k_ObservationStacksName), true);
            }
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.PropertyField(so.FindProperty(k_CompressionTypeName), true);

            EditorGUILayout.LabelField("Debug Gizmo", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(so.FindProperty(k_ShowGizmosName), true);
            EditorGUILayout.PropertyField(so.FindProperty(k_GizmoYOffsetName), true);

            // detectable objects
            var debugColors = so.FindProperty("m_DebugColors");
            var detectableObjectSize = so.FindProperty(k_DetectableObjectsName).arraySize;
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
