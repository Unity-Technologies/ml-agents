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
        const string k_ActionSpecName = "m_ActionSpec";
        const string k_ContinuousActionSizeName = "m_NumContinuousActions";
        const string k_DiscreteBranchSizeName = "BranchSizes";
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
            EditorGUI.LabelField(position, "Actions");
            position.y += k_LineHeight;
            EditorGUI.indentLevel++;
            var actionSpecProperty = property.FindPropertyRelative(k_ActionSpecName);
            DrawContinuousVectorAction(position, actionSpecProperty);
            position.y += k_LineHeight;
            DrawDiscreteVectorAction(position, actionSpecProperty);
        }

        /// <summary>
        /// Draws the Continuous Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawContinuousVectorAction(Rect position, SerializedProperty property)
        {
            var continuousActionSize = property.FindPropertyRelative(k_ContinuousActionSizeName);
            EditorGUI.PropertyField(
                position,
                continuousActionSize,
                new GUIContent("Continuous Actions", "Number of continuous actions."));
        }

        /// <summary>
        /// Draws the Discrete Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        static void DrawDiscreteVectorAction(Rect position, SerializedProperty property)
        {
            var branchSizes = property.FindPropertyRelative(k_DiscreteBranchSizeName);
            var newSize = EditorGUI.IntField(
                position, "Discrete Branches", branchSizes.arraySize);

            // This check is here due to:
            // https://fogbugz.unity3d.com/f/cases/1246524/
            // If this case has been resolved, please remove this if condition.
            if (newSize != branchSizes.arraySize)
            {
                branchSizes.arraySize = newSize;
            }

            position.y += k_LineHeight;
            position.x += 20;
            position.width -= 20;
            for (var branchIndex = 0;
                 branchIndex < branchSizes.arraySize;
                 branchIndex++)
            {
                var branchActionSize =
                    branchSizes.GetArrayElementAtIndex(branchIndex);

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
            var actionSpecProperty = property.FindPropertyRelative(k_ActionSpecName);
            var numActionLines = 3 + actionSpecProperty.FindPropertyRelative(k_DiscreteBranchSizeName).arraySize;
            return numActionLines * k_LineHeight;
        }
    }
}
