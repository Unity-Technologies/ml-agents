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
        private const float LineHeight = 17f;
        private const int VecObsNumLine = 3;
        private const string CamResPropName = "cameraResolutions";
        private const string ActionSizePropName = "vectorActionSize";
        private const string ActionTypePropName = "vectorActionSpaceType";
        private const string ActionDescriptionPropName = "vectorActionDescriptions";
        private const string VecObsPropName = "vectorObservationSize";
        private const string NumVecObsPropName ="numStackedVectorObservations";
        private const string CamWidthPropName = "width";
        private const string CamHeightPropName = "height";
        private const string CamGrayPropName = "blackAndWhite";
        private const int DefaultCameraWidth = 84;
        private const int DefaultCameraHeight = 84;
        private const bool DefaultCameraGray = false;
        
        /// <inheritdoc />
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (property.isExpanded)
            {
                return LineHeight + 
                       GetHeightDrawVectorObservation() +
                       GetHeightDrawVisualObservation(property) +
                       GetHeightDrawVectorAction(property) +
                       GetHeightDrawVectorActionDescriptions(property);
            }
            return LineHeight;
        }

        /// <inheritdoc />
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            position.height = LineHeight;
            property.isExpanded = EditorGUI.Foldout(position, property.isExpanded, label);
            position.y += LineHeight;
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
            position.y += LineHeight;

            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(VecObsPropName),
                new GUIContent("Space Size",
                    "Length of state " +
                    "vector for brain (In Continuous state space)." +
                    "Or number of possible values (in Discrete state space)."));
            position.y += LineHeight;
            
            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(NumVecObsPropName),
                new GUIContent("Stacked Vectors",
                    "Number of states that will be stacked before " +
                    "beeing fed to the neural network."));
            position.y += LineHeight;
            EditorGUI.indentLevel--; 
        }

        /// <summary>
        /// The Height required to draw the Vector Observations paramaters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Observations </returns>
        private static float GetHeightDrawVectorObservation()
        {
            return VecObsNumLine * LineHeight;
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
            position.y += LineHeight;
            var quarter = position.width / 4;
            var resolutions = property.FindPropertyRelative(CamResPropName);
            DrawVisualObsButtons(position, resolutions);
            position.y += LineHeight;
            
            // Display the labels for the columns : Index, Width, Height and Gray
            var indexRect = new Rect(position.x, position.y, quarter, position.height);
            var widthRect = new Rect(position.x + quarter, position.y, quarter, position.height);
            var heightRect = new Rect(position.x + 2*quarter, position.y, quarter, position.height);
            var bwRect = new Rect(position.x + 3*quarter, position.y, quarter, position.height);
            EditorGUI.indentLevel++;
            if (resolutions.arraySize > 0)
            {
                EditorGUI.LabelField(indexRect, "Index");
                indexRect.y += LineHeight;
                EditorGUI.LabelField(widthRect, "Width");
                widthRect.y += LineHeight;
                EditorGUI.LabelField(heightRect, "Height");
                heightRect.y += LineHeight;
                EditorGUI.LabelField(bwRect, "Gray");
                bwRect.y += LineHeight;
            }

            // Iterate over the resolutions
            for (var i = 0; i < resolutions.arraySize; i++)
            {
                EditorGUI.LabelField(indexRect, "Obs " + i);
                indexRect.y += LineHeight;
                var res = resolutions.GetArrayElementAtIndex(i);
                var w = res.FindPropertyRelative("width");
                w.intValue = EditorGUI.IntField(widthRect, w.intValue);
                widthRect.y += LineHeight;
                var h = res.FindPropertyRelative("height");
                h.intValue = EditorGUI.IntField(heightRect, h.intValue);
                heightRect.y += LineHeight;
                var bw = res.FindPropertyRelative("blackAndWhite");
                bw.boolValue = EditorGUI.Toggle(bwRect, bw.boolValue);
                bwRect.y += LineHeight;
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
                newRes.FindPropertyRelative(CamWidthPropName).intValue = DefaultCameraWidth;
                newRes.FindPropertyRelative(CamHeightPropName).intValue = DefaultCameraHeight;
                newRes.FindPropertyRelative(CamGrayPropName).boolValue = DefaultCameraGray;

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
            var visObsSize = property.FindPropertyRelative(CamResPropName).arraySize + 2;
            if (property.FindPropertyRelative(CamResPropName).arraySize > 0)
            {
                visObsSize += 1;
            }
            return LineHeight * visObsSize;
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
            position.y += LineHeight;
            EditorGUI.indentLevel++;
            var bpVectorActionType = property.FindPropertyRelative(ActionTypePropName);
            EditorGUI.PropertyField(
                position,
                bpVectorActionType,
                new GUIContent("Space Type",
                    "Corresponds to whether state vector contains  a single integer (Discrete) " +
                    "or a series of real-valued floats (Continuous)."));
            position.y += LineHeight;
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
            var vecActionSize = property.FindPropertyRelative(ActionSizePropName);
            vecActionSize.arraySize = 1;
            SerializedProperty continuousActionSize =
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
            var vecActionSize = property.FindPropertyRelative(ActionSizePropName);
            vecActionSize.arraySize = EditorGUI.IntField(
                position, "Branches Size", vecActionSize.arraySize);
            position.y += LineHeight;
            position.x += 20;
            position.width -= 20;
            for (var branchIndex = 0;
                branchIndex < vecActionSize.arraySize;
                branchIndex++)
            {
                SerializedProperty branchActionSize =
                    vecActionSize.GetArrayElementAtIndex(branchIndex);
                
                EditorGUI.PropertyField(
                    position,
                    branchActionSize,
                    new GUIContent("Branch " + branchIndex + " Size",
                        "Number of possible actions for the branch number " + branchIndex + "."));
                position.y += LineHeight;
            }
        }
        
        /// <summary>
        /// The Height required to draw the Vector Action parameters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Action </returns>
        private static float GetHeightDrawVectorAction(SerializedProperty property)
        {
            var actionSize = 2 + property.FindPropertyRelative(ActionSizePropName).arraySize;
            if (property.FindPropertyRelative(ActionTypePropName).enumValueIndex == 0)
            {
                actionSize += 1;
            }
            return actionSize * LineHeight;
        }

        /// <summary>
        /// Draws the Vector Actions descriptions for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private static void DrawVectorActionDescriptions(Rect position, SerializedProperty property)
        {
            var bpVectorActionType = property.FindPropertyRelative(ActionTypePropName);
            var vecActionSize = property.FindPropertyRelative(ActionSizePropName);
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
                property.FindPropertyRelative(ActionDescriptionPropName);
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
                position.y += LineHeight;
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
                position.y += LineHeight;
            }
        }
        /// <summary>
        /// The Height required to draw the Action Descriptions
        /// </summary>
        /// <returns>The height of the drawer of the Action Descriptions </returns>
        private static float GetHeightDrawVectorActionDescriptions(SerializedProperty property)
        {
            var descriptionSize = 1;
            if (property.FindPropertyRelative(ActionDescriptionPropName).isExpanded)
            {
                var descriptions = property.FindPropertyRelative(ActionDescriptionPropName);
                descriptionSize += descriptions.arraySize + 1;
            }
            return descriptionSize * LineHeight;
        }
    }
}
