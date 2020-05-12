using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Editor
{
    /// <summary>
    /// PropertyDrawer for BrainParameters. Defines how BrainParameters are displayed in the
    /// Inspector.
    /// </summary>
    [CustomPropertyDrawer(typeof(BrainParameters))]
    internal class BrainParametersDrawer : PropertyDrawer
    {
        // The height of a line in the Unity Inspectors
        const float k_LineHeight = 17f;
        const int k_VecObsNumLine = 3;
        const string k_ActionSizePropName = "VectorActionSize";
        const string k_ActionTypePropName = "VectorActionSpaceType";
        const string k_ActionDescriptionPropName = "VectorActionDescriptions";
        const string k_VecObsPropName = "VectorObservationSize";
        const string k_NumVecObsPropName = "NumStackedVectorObservations";

        /// <inheritdoc />
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return GetHeightDrawVectorObservation() +
                GetHeightDrawVectorAction(property);
        }

        /// <inheritdoc />
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            position.height = k_LineHeight;
            EditorGUI.BeginProperty(position, label, property);
            EditorGUI.indentLevel++;

            // Vector Observations
            DrawVectorObservation(position, property);
            position.y += GetHeightDrawVectorObservation();

            // Vector Action
            DrawVectorAction(position, property);
            position.y += GetHeightDrawVectorAction(property);

            EditorGUI.EndProperty();
            EditorGUI.indentLevel = indent;
        }

        /// <summary>
        /// Draws the Vector Observations for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawVectorObservation(Rect position, SerializedProperty property)
        {
            EditorGUI.LabelField(position, "Vector Observation");
            position.y += k_LineHeight;

            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(k_VecObsPropName),
                new GUIContent("Space Size",
                    "Length of state " +
                    "vector for brain (In Continuous state space)." +
                    "Or number of possible values (in Discrete state space)."));
            position.y += k_LineHeight;

            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(k_NumVecObsPropName),
                new GUIContent("Stacked Vectors",
                    "Number of states that will be stacked before " +
                    "being fed to the neural network."));
            position.y += k_LineHeight;
            EditorGUI.indentLevel--;
        }

        /// <summary>
        /// The Height required to draw the Vector Observations paramaters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Observations </returns>
        static float GetHeightDrawVectorObservation()
        {
            return k_VecObsNumLine * k_LineHeight;
        }

        /// <summary>
        /// Draws the Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawVectorAction(Rect position, SerializedProperty property)
        {
            EditorGUI.LabelField(position, "Vector Action");
            position.y += k_LineHeight;
            EditorGUI.indentLevel++;
            var bpVectorActionType = property.FindPropertyRelative(k_ActionTypePropName);
            EditorGUI.PropertyField(
                position,
                bpVectorActionType,
                new GUIContent("Space Type",
                    "Corresponds to whether state vector contains  a single integer (Discrete) " +
                    "or a series of real-valued floats (Continuous)."));
            position.y += k_LineHeight;
            if (bpVectorActionType.enumValueIndex == 1)
            {
                DrawContinuousVectorAction(position, property);
            }
            else
            {
                DrawDiscreteVectorAction(position, property);
            }
        }

        /// <summary>
        /// Draws the Continuous Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawContinuousVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(k_ActionSizePropName);

            // This check is here due to:
            // https://fogbugz.unity3d.com/f/cases/1246524/
            // If this case has been resolved, please remove this if condition.
            if (vecActionSize.arraySize != 1)
            {
                vecActionSize.arraySize = 1;
            }
            var continuousActionSize =
                vecActionSize.GetArrayElementAtIndex(0);
            EditorGUI.PropertyField(
                position,
                continuousActionSize,
                new GUIContent("Space Size", "Length of continuous action vector."));
        }

        /// <summary>
        /// Draws the Discrete Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawDiscreteVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(k_ActionSizePropName);
            var newSize = EditorGUI.IntField(
                position, "Branches Size", vecActionSize.arraySize);

            // This check is here due to:
            // https://fogbugz.unity3d.com/f/cases/1246524/
            // If this case has been resolved, please remove this if condition.
            if (newSize != vecActionSize.arraySize)
            {
                vecActionSize.arraySize = newSize;
            }

            position.y += k_LineHeight;
            position.x += 20;
            position.width -= 20;
            for (var branchIndex = 0;
                 branchIndex < vecActionSize.arraySize;
                 branchIndex++)
            {
                var branchActionSize =
                    vecActionSize.GetArrayElementAtIndex(branchIndex);

                EditorGUI.PropertyField(
                    position,
                    branchActionSize,
                    new GUIContent("Branch " + branchIndex + " Size",
                        "Number of possible actions for the branch number " + branchIndex + "."));
                position.y += k_LineHeight;
            }
        }

        /// <summary>
        /// The Height required to draw the Vector Action parameters.
        /// </summary>
        /// <returns>The height of the drawer of the Vector Action.</returns>
        static float GetHeightDrawVectorAction(SerializedProperty property)
        {
            var actionSize = 2 + property.FindPropertyRelative(k_ActionSizePropName).arraySize;
            if (property.FindPropertyRelative(k_ActionTypePropName).enumValueIndex == 0)
            {
                actionSize += 1;
            }
            return actionSize * k_LineHeight;
        }
    }
}
