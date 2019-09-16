using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /// <summary>
    /// PropertyDrawer for BrainParameters. Defines how BrainParameters are displayed in the
    /// Inspector.
    /// </summary>
    [CustomPropertyDrawer(typeof(BrainParameters))]
    public class BrainParametersDrawer : PropertyDrawer
    {
        // The height of a line in the Unity Inspectors
        private const float k_LineHeight = 17f;
        private const int k_VecObsNumLine = 3;
        private const string k_CamResPropName = "cameraResolutions";
        private const string k_ActionSizePropName = "vectorActionSize";
        private const string k_ActionTypePropName = "vectorActionSpaceType";
        private const string k_ActionDescriptionPropName = "vectorActionDescriptions";
        private const string k_VecObsPropName = "vectorObservationSize";
        private const string k_NumVecObsPropName = "numStackedVectorObservations";
        private const string k_CamWidthPropName = "width";
        private const string k_CamHeightPropName = "height";
        private const string k_CamGrayPropName = "blackAndWhite";
        private const int k_DefaultCameraWidth = 84;
        private const int k_DefaultCameraHeight = 84;
        private const bool k_DefaultCameraGray = false;

        /// <inheritdoc />
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (property.isExpanded)
            {
                return k_LineHeight +
                    GetHeightDrawVectorObservation() +
                    GetHeightDrawVisualObservation(property) +
                    GetHeightDrawVectorAction(property) +
                    GetHeightDrawVectorActionDescriptions(property);
            }
            return k_LineHeight;
        }

        /// <inheritdoc />
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            position.height = k_LineHeight;
            property.isExpanded = EditorGUI.Foldout(position, property.isExpanded, label);
            position.y += k_LineHeight;
            if (property.isExpanded)
            {
                EditorGUI.BeginProperty(position, label, property);
                EditorGUI.indentLevel++;

                // Vector Observations
                DrawVectorObservation(position, property);
                position.y += GetHeightDrawVectorObservation();

                //Visual Observations
                DrawVisualObservations(position, property);
                position.y += GetHeightDrawVisualObservation(property);

                // Vector Action
                DrawVectorAction(position, property);
                position.y += GetHeightDrawVectorAction(property);

                // Vector Action Descriptions
                DrawVectorActionDescriptions(position, property);
                position.y += GetHeightDrawVectorActionDescriptions(property);
                EditorGUI.EndProperty();
            }
            EditorGUI.indentLevel = indent;
        }

        /// <summary>
        /// Draws the Vector Observations for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private static void DrawVectorObservation(Rect position, SerializedProperty property)
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
        private static float GetHeightDrawVectorObservation()
        {
            return k_VecObsNumLine * k_LineHeight;
        }

        /// <summary>
        /// Draws the Visual Observations parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private static void DrawVisualObservations(Rect position, SerializedProperty property)
        {
            EditorGUI.LabelField(position, "Visual Observations");
            position.y += k_LineHeight;
            var quarter = position.width / 4;
            var resolutions = property.FindPropertyRelative(k_CamResPropName);
            DrawVisualObsButtons(position, resolutions);
            position.y += k_LineHeight;

            // Display the labels for the columns : Index, Width, Height and Gray
            var indexRect = new Rect(position.x, position.y, quarter, position.height);
            var widthRect = new Rect(position.x + quarter, position.y, quarter, position.height);
            var heightRect = new Rect(position.x + 2 * quarter, position.y, quarter, position.height);
            var bwRect = new Rect(position.x + 3 * quarter, position.y, quarter, position.height);
            EditorGUI.indentLevel++;
            if (resolutions.arraySize > 0)
            {
                EditorGUI.LabelField(indexRect, "Index");
                indexRect.y += k_LineHeight;
                EditorGUI.LabelField(widthRect, "Width");
                widthRect.y += k_LineHeight;
                EditorGUI.LabelField(heightRect, "Height");
                heightRect.y += k_LineHeight;
                EditorGUI.LabelField(bwRect, "Gray");
                bwRect.y += k_LineHeight;
            }

            // Iterate over the resolutions
            for (var i = 0; i < resolutions.arraySize; i++)
            {
                EditorGUI.LabelField(indexRect, "Obs " + i);
                indexRect.y += k_LineHeight;
                var res = resolutions.GetArrayElementAtIndex(i);
                var w = res.FindPropertyRelative("width");
                w.intValue = EditorGUI.IntField(widthRect, w.intValue);
                widthRect.y += k_LineHeight;
                var h = res.FindPropertyRelative("height");
                h.intValue = EditorGUI.IntField(heightRect, h.intValue);
                heightRect.y += k_LineHeight;
                var bw = res.FindPropertyRelative("blackAndWhite");
                bw.boolValue = EditorGUI.Toggle(bwRect, bw.boolValue);
                bwRect.y += k_LineHeight;
            }
            EditorGUI.indentLevel--;
        }

        /// <summary>
        /// Draws the buttons to add and remove the visual observations parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="resolutions">The SerializedProperty of the resolution array
        /// to make the custom GUI for.</param>
        private static void DrawVisualObsButtons(Rect position, SerializedProperty resolutions)
        {
            var widthEighth = position.width / 8;
            var addButtonRect = new Rect(position.x + widthEighth, position.y,
                3 * widthEighth, position.height);
            var removeButtonRect = new Rect(position.x + 4 * widthEighth, position.y,
                3 * widthEighth, position.height);
            if (resolutions.arraySize == 0)
            {
                addButtonRect.width *= 2;
            }
            // Display the buttons
            if (GUI.Button(addButtonRect, "Add New", EditorStyles.miniButton))
            {
                resolutions.arraySize += 1;
                var newRes = resolutions.GetArrayElementAtIndex(resolutions.arraySize - 1);
                newRes.FindPropertyRelative(k_CamWidthPropName).intValue = k_DefaultCameraWidth;
                newRes.FindPropertyRelative(k_CamHeightPropName).intValue = k_DefaultCameraHeight;
                newRes.FindPropertyRelative(k_CamGrayPropName).boolValue = k_DefaultCameraGray;
            }
            if (resolutions.arraySize > 0)
            {
                if (GUI.Button(removeButtonRect, "Remove Last", EditorStyles.miniButton))
                {
                    resolutions.arraySize -= 1;
                }
            }
        }

        /// <summary>
        /// The Height required to draw the Visual Observations parameters
        /// </summary>
        /// <returns>The height of the drawer of the Visual Observations </returns>
        private static float GetHeightDrawVisualObservation(SerializedProperty property)
        {
            var visObsSize = property.FindPropertyRelative(k_CamResPropName).arraySize + 2;
            if (property.FindPropertyRelative(k_CamResPropName).arraySize > 0)
            {
                visObsSize += 1;
            }
            return k_LineHeight * visObsSize;
        }

        /// <summary>
        /// Draws the Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private static void DrawVectorAction(Rect position, SerializedProperty property)
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
        private static void DrawContinuousVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(k_ActionSizePropName);
            vecActionSize.arraySize = 1;
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
        private static void DrawDiscreteVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(k_ActionSizePropName);
            vecActionSize.arraySize = EditorGUI.IntField(
                position, "Branches Size", vecActionSize.arraySize);
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
        /// The Height required to draw the Vector Action parameters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Action </returns>
        private static float GetHeightDrawVectorAction(SerializedProperty property)
        {
            var actionSize = 2 + property.FindPropertyRelative(k_ActionSizePropName).arraySize;
            if (property.FindPropertyRelative(k_ActionTypePropName).enumValueIndex == 0)
            {
                actionSize += 1;
            }
            return actionSize * k_LineHeight;
        }

        /// <summary>
        /// Draws the Vector Actions descriptions for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private static void DrawVectorActionDescriptions(Rect position, SerializedProperty property)
        {
            var bpVectorActionType = property.FindPropertyRelative(k_ActionTypePropName);
            var vecActionSize = property.FindPropertyRelative(k_ActionSizePropName);
            var numberOfDescriptions = 0;
            if (bpVectorActionType.enumValueIndex == 1)
            {
                numberOfDescriptions = vecActionSize.GetArrayElementAtIndex(0).intValue;
            }
            else
            {
                numberOfDescriptions = vecActionSize.arraySize;
            }

            EditorGUI.indentLevel++;
            var vecActionDescriptions =
                property.FindPropertyRelative(k_ActionDescriptionPropName);
            vecActionDescriptions.arraySize = numberOfDescriptions;
            if (bpVectorActionType.enumValueIndex == 1)
            {
                //Continuous case :
                EditorGUI.PropertyField(
                    position,
                    vecActionDescriptions,
                    new GUIContent("Action Descriptions",
                        "A list of strings used to name the available actionsm for the Brain."),
                    true);
                position.y += k_LineHeight;
            }
            else
            {
                // Discrete case :
                EditorGUI.PropertyField(
                    position,
                    vecActionDescriptions,
                    new GUIContent("Branch Descriptions",
                        "A list of strings used to name the available branches for the Brain."),
                    true);
                position.y += k_LineHeight;
            }
        }

        /// <summary>
        /// The Height required to draw the Action Descriptions
        /// </summary>
        /// <returns>The height of the drawer of the Action Descriptions </returns>
        private static float GetHeightDrawVectorActionDescriptions(SerializedProperty property)
        {
            var descriptionSize = 1;
            if (property.FindPropertyRelative(k_ActionDescriptionPropName).isExpanded)
            {
                var descriptions = property.FindPropertyRelative(k_ActionDescriptionPropName);
                descriptionSize += descriptions.arraySize + 1;
            }
            return descriptionSize * k_LineHeight;
        }
    }
}
