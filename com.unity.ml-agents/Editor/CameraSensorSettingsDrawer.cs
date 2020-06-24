using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEngine;

namespace Unity.MLAgents.Editor
{
    /// <summary>
    /// PropertyDrawer for BrainParameters. Defines how BrainParameters are displayed in the
    /// Inspector.
    /// </summary>
    [CustomPropertyDrawer(typeof(CameraSensorSettings))]
    internal class CameraSensorSettingsDrawer : PropertyDrawer
    {
        // The height of a line in the Unity Inspectors
        const float k_LineHeight = 17f;
        const string k_LayerMasksPropName = "LayerMasks";
        const string k_DisableCameraPropName = "DisableCamera";
        const string k_EnableDepthPropName = "EnableDepth";
        const string k_EnableAutoSegmentPropName = "EnableAutoSegment";

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return GetHeightDrawLayerMasks(property) + 4 * k_LineHeight;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var indent = EditorGUI.indentLevel;
            position.height = k_LineHeight;
            EditorGUI.BeginProperty(position, label, property);

            var disableCamProp = property.FindPropertyRelative(k_DisableCameraPropName);
            EditorGUI.PropertyField(position, disableCamProp, new GUIContent("Disable camera"));
            position.y += k_LineHeight;

            var enableDepthProp = property.FindPropertyRelative(k_EnableDepthPropName);
            EditorGUI.PropertyField(position, enableDepthProp, new GUIContent("Enable depth channel"));
            position.y += k_LineHeight;

            var enableAutoSegmentProp = property.FindPropertyRelative(k_EnableAutoSegmentPropName);
            EditorGUI.PropertyField(position, enableAutoSegmentProp, new GUIContent("Enable auto segment"));
            position.y += k_LineHeight;

            // Vector Observations
            DrawLayerMasks(position, property);
            position.y += GetHeightDrawLayerMasks(property);

            EditorGUI.EndProperty();
            EditorGUI.indentLevel = indent;
        }

        /// <summary>
        /// Draws the Layer Masks for the CameraSensorComponent
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the CameraSensorComponent
        /// to make the custom GUI for.</param>
        static void DrawLayerMasks(Rect position, SerializedProperty property)
        {
            var layerMasksProp = property.FindPropertyRelative(k_LayerMasksPropName);
            var newSize = EditorGUI.IntField(
                position,
                "# Layer Masks",
                layerMasksProp.arraySize
            );

            // This check is here due to:
            // https://fogbugz.unity3d.com/f/cases/1246524/
            // If this case has been resolved, please remove this if condition.
            if (newSize != layerMasksProp.arraySize)
            {
                layerMasksProp.arraySize = newSize;
            }

            position.y += k_LineHeight;
            position.x += 20;
            position.width -= 20;
            for (var branchIndex = 0;
                branchIndex < layerMasksProp.arraySize;
                branchIndex++)
            {
                var layerMask =
                    layerMasksProp.GetArrayElementAtIndex(branchIndex);

                var newMaskLayer = EditorGUI.LayerField(
                    position,
                    new GUIContent("Mask " + branchIndex + " layer number:",
                        "Layer number for layer mask " + branchIndex + "."),
                    layerMask.intValue
                );
                if (layerMask.intValue != newMaskLayer)
                {
                    layerMask.intValue = newMaskLayer;
                }
                position.y += k_LineHeight;
            }
        }

        static float GetHeightDrawLayerMasks(SerializedProperty property)
        {
            var layerMasksProp = property.FindPropertyRelative(k_LayerMasksPropName);
            return (1 + layerMasksProp.arraySize) * k_LineHeight;
        }
    }
}
