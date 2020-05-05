using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Editor
{
    internal class RayPerceptionSensorComponentBaseEditor : UnityEditor.Editor
    {
        bool m_RequireSensorUpdate;

        protected void OnRayPerceptionInspectorGUI(bool is3d)
        {
            var so = serializedObject;
            so.Update();

            // Drawing the RayPerceptionSensorComponent
            EditorGUI.BeginChangeCheck();
            EditorGUI.indentLevel++;

            // Don't allow certain fields to be modified during play mode.
            // * SensorName affects the ordering of the Agent's observations
            // * The number of tags and rays affects the size of the observations.
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_DetectableTags"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_RaysPerDirection"), true);
            }
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.PropertyField(so.FindProperty("m_MaxRayDegrees"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_SphereCastRadius"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_RayLength"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_RayLayerMask"), true);

            // Because the number of observation stacks affects the observation shape,
            // it is not editable during play mode.
            EditorGUI.BeginDisabledGroup(!EditorUtilities.CanUpdateModelProperties());
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_ObservationStacks"), new GUIContent("Stacked Raycasts"), true);
            }
            EditorGUI.EndDisabledGroup();

            if (is3d)
            {
                EditorGUILayout.PropertyField(so.FindProperty("m_StartVerticalOffset"), true);
                EditorGUILayout.PropertyField(so.FindProperty("m_EndVerticalOffset"), true);
            }

            EditorGUILayout.PropertyField(so.FindProperty("rayHitColor"), true);
            EditorGUILayout.PropertyField(so.FindProperty("rayMissColor"), true);

            EditorGUI.indentLevel--;
            if (EditorGUI.EndChangeCheck())
            {
                m_RequireSensorUpdate = true;
            }

            so.ApplyModifiedProperties();
            UpdateSensorIfDirty();
        }

        void UpdateSensorIfDirty()
        {
            if (m_RequireSensorUpdate)
            {
                var sensorComponent = serializedObject.targetObject as RayPerceptionSensorComponentBase;
                sensorComponent?.UpdateSensor();
                m_RequireSensorUpdate = false;
            }
        }
    }

    [CustomEditor(typeof(RayPerceptionSensorComponent2D))]
    [CanEditMultipleObjects]
    internal class RayPerceptionSensorComponent2DEditor : RayPerceptionSensorComponentBaseEditor
    {
        public override void OnInspectorGUI()
        {
            OnRayPerceptionInspectorGUI(false);
        }
    }

    [CustomEditor(typeof(RayPerceptionSensorComponent3D))]
    [CanEditMultipleObjects]
    internal class RayPerceptionSensorComponent3DEditor : RayPerceptionSensorComponentBaseEditor
    {
        public override void OnInspectorGUI()
        {
            OnRayPerceptionInspectorGUI(true);
        }
    }
}
