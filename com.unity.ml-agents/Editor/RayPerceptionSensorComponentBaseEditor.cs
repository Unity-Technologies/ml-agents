using UnityEngine;
using UnityEditor;
using Barracuda;

namespace MLAgents
{
    //[CustomEditor(typeof(RayPerceptionSensorComponentBase))]
    //[CustomEditor(typeof(RayPerceptionSensorComponent2D))]  # TODO handle inheritance
    [CustomEditor(typeof(RayPerceptionSensorComponent3D))]
    [CanEditMultipleObjects]
    internal class RayPerceptionSensorComponentBaseEditor : Editor
    {
        bool m_RequireSensorUpdate;

        public override void OnInspectorGUI()
        {
            var so = serializedObject;
            so.Update();

            // Drawing the Behavior Parameters
            EditorGUI.BeginChangeCheck();
            EditorGUI.indentLevel++;

            EditorGUILayout.PropertyField(so.FindProperty("m_SensorName"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_DetectableTags"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_RaysPerDirection"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_MaxRayDegrees"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_SphereCastRadius"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_RayLength"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_RayLayerMask"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_ObservationStacks"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_StartVerticalOffset"), true);
            EditorGUILayout.PropertyField(so.FindProperty("m_EndVerticalOffset"), true);

            EditorGUILayout.PropertyField(so.FindProperty("rayHitColor"), true);
            EditorGUILayout.PropertyField(so.FindProperty("rayMissColor"), true);

            EditorGUI.indentLevel--;
            if (EditorGUI.EndChangeCheck())
            {
                m_RequireSensorUpdate = true;
            }

            UpdateSensorIfDirty();
            so.ApplyModifiedProperties();
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
}
