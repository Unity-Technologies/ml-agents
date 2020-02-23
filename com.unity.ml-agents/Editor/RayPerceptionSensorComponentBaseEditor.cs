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

            EditorGUILayout.PropertyField(so.FindProperty("sensorName"), true);
            EditorGUILayout.PropertyField(so.FindProperty("detectableTags"), true);
            EditorGUILayout.PropertyField(so.FindProperty("raysPerDirection"), true);
            EditorGUILayout.PropertyField(so.FindProperty("maxRayDegrees"), true);
            EditorGUILayout.PropertyField(so.FindProperty("sphereCastRadius"), true);
            EditorGUILayout.PropertyField(so.FindProperty("rayLength"), true);
            EditorGUILayout.PropertyField(so.FindProperty("rayLayerMask"), true);
            EditorGUILayout.PropertyField(so.FindProperty("observationStacks"), true);
            EditorGUILayout.PropertyField(so.FindProperty("startVerticalOffset"), true);
            EditorGUILayout.PropertyField(so.FindProperty("endVerticalOffset"), true);

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
