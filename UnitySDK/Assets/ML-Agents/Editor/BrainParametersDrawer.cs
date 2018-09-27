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
        private const float lineHeight = 17f;
        private const string camResPropName = "cameraResolutions";
        private const string actionSizePropName = "vectorActionSize";
        private const string actionTypePropName = "vectorActionSpaceType";
        private const string actionDescriptionPropName = "vectorActionDescriptions";
        private const string vecObsPropName = "vectorObservationSize";
        private const string numVecObsPropName ="numStackedVectorObservations";
        private const string camWidthPropName = "width";
        private const string camHeightPropName = "height";
        private const string camGrayPropName = "blackAndWhite";
        private const int defaultCameraWidth = 84;
        private const int defaultCameraHeight = 84;
        private const bool defaultCameraGray = false;
        
        /// <summary>
        /// Computes the height of the Drawer depending on the property it is showing
        /// </summary>
        /// <param name="property">The property that is being drawn.</param>
        /// <param name="label">The label of the property being drawn.</param>
        /// <returns>The vertical space needed to draw the property.</returns>
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (property.isExpanded)
            {
                return lineHeight + 
                       GetHeightDrawVectorObservation() +
                       GetHeightDrawVisualObservation(property) +
                       GetHeightDrawVectorAction(property) +
                       GetHeightDrawVectorActionDescriptions(property);
            }
            return lineHeight;
        }

        /// <summary>
        /// Draws the Brain Parameter property
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        /// <param name="label">The label of this property.</param>
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            position.height = lineHeight;
            property.isExpanded = EditorGUI.Foldout(position, property.isExpanded, label);
            position.y += lineHeight;
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
        private void DrawVectorObservation(Rect position, SerializedProperty property)
        {
            EditorGUI.LabelField(position, "Vector Observation");
            position.y += lineHeight;

            EditorGUI.indentLevel++;
            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(vecObsPropName),
                new GUIContent("Space Size",
                    "Length of state " +
                    "vector for brain (In Continuous state space)." +
                    "Or number of possible values (in Discrete state space)."));
            position.y += lineHeight;
            
            EditorGUI.PropertyField(position,
                property.FindPropertyRelative(numVecObsPropName),
                new GUIContent("Stacked Vectors",
                    "Number of states that will be stacked before " +
                    "beeing fed to the neural network."));
            position.y += lineHeight;
            EditorGUI.indentLevel--; 
        }

        /// <summary>
        /// The Height required to draw the Vector Observations paramaters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Observations </returns>
        private float GetHeightDrawVectorObservation()
        {
            return 3 * lineHeight;
        }

        /// <summary>
        /// Draws the Visual Observations parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private void DrawVisualObservations(Rect position, SerializedProperty property)
        {            
            EditorGUI.LabelField(position, "Visual Observations");
            position.y += lineHeight;
            var quarter = position.width / 4;
            var resolutions = property.FindPropertyRelative(camResPropName);
            DrawVisualObservationButtons(position, resolutions);
            position.y += lineHeight;
            
            // Display the labels for the columns : Index, Width, Height and Gray
            var indexRect = new Rect(position.x, position.y, quarter, position.height);
            var widthRect = new Rect(position.x + quarter, position.y, quarter, position.height);
            var heightRect = new Rect(position.x + 2*quarter, position.y, quarter, position.height);
            var bwRect = new Rect(position.x + 3*quarter, position.y, quarter, position.height);
            EditorGUI.indentLevel++;
            if (resolutions.arraySize > 0)
            {
                EditorGUI.LabelField(indexRect, "Index");
                indexRect.y += lineHeight;
                EditorGUI.LabelField(widthRect, "Width");
                widthRect.y += lineHeight;
                EditorGUI.LabelField(heightRect, "Height");
                heightRect.y += lineHeight;
                EditorGUI.LabelField(bwRect, "Gray");
                bwRect.y += lineHeight;
            }

            // Iterate over the resolutions
            for (var i = 0; i < resolutions.arraySize; i++)
            {
                EditorGUI.LabelField(indexRect, "Obs " + i);
                indexRect.y += lineHeight;
                var res = resolutions.GetArrayElementAtIndex(i);
                var w = res.FindPropertyRelative("width");
                w.intValue = EditorGUI.IntField(widthRect, w.intValue);
                widthRect.y += lineHeight;
                var h = res.FindPropertyRelative("height");
                h.intValue = EditorGUI.IntField(heightRect, h.intValue);
                heightRect.y += lineHeight;
                var bw = res.FindPropertyRelative("blackAndWhite");
                bw.boolValue = EditorGUI.Toggle(bwRect, bw.boolValue);
                bwRect.y += lineHeight;
            }
            EditorGUI.indentLevel--;
        }

        /// <summary>
        /// Draws the buttons to add and remove the visual observations parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="resolutions">The SerializedProperty of the resolution array
        /// to make the custom GUI for.</param>
        private void DrawVisualObservationButtons(Rect position, SerializedProperty resolutions)
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
                newRes.FindPropertyRelative(camWidthPropName).intValue = defaultCameraWidth;
                newRes.FindPropertyRelative(camHeightPropName).intValue = defaultCameraHeight;
                newRes.FindPropertyRelative(camGrayPropName).boolValue = defaultCameraGray;

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
        private float GetHeightDrawVisualObservation(SerializedProperty property)
        {
            var visObsSize = property.FindPropertyRelative(camResPropName).arraySize + 2;
            if (property.FindPropertyRelative(camResPropName).arraySize > 0)
            {
                visObsSize += 1;
            }
            return lineHeight * visObsSize;
        }

        /// <summary>
        /// Draws the Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private void DrawVectorAction(Rect position, SerializedProperty property)
        {
            EditorGUI.LabelField(position, "Vector Action"); 
            position.y += lineHeight;
            EditorGUI.indentLevel++;
            var bpVectorActionType = property.FindPropertyRelative(actionTypePropName);
            EditorGUI.PropertyField(
                position,
                bpVectorActionType,
                new GUIContent("Space Type",
                    "Corresponds to whether state vector contains  a single integer (Discrete) " +
                    "or a series of real-valued floats (Continuous)."));
            position.y += lineHeight;
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
        private void DrawContinuousVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(actionSizePropName);
            vecActionSize.arraySize = 1;
            SerializedProperty continuousActionSize =
                vecActionSize.GetArrayElementAtIndex(0);
            EditorGUI.PropertyField(
                position,
                continuousActionSize,
                new GUIContent("Space Size",
                    "Length of continuous action vector."));
        }
        
        /// <summary>
        /// Draws the Discrete Vector Actions parameters for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private void DrawDiscreteVectorAction(Rect position, SerializedProperty property)
        {
            var vecActionSize = property.FindPropertyRelative(actionSizePropName);
            vecActionSize.arraySize = EditorGUI.IntField(
                position,
                "Branches Size",
                vecActionSize.arraySize);
            position.y += lineHeight;
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
                    new GUIContent(
                        "Branch " + branchIndex + " Size",
                        "Number of possible actions for the " +
                        "branch number " + branchIndex + "."));
                position.y += lineHeight;
            }
        }
        
        /// <summary>
        /// The Height required to draw the Vector Action parameters
        /// </summary>
        /// <returns>The height of the drawer of the Vector Action </returns>
        private float GetHeightDrawVectorAction(SerializedProperty property)
        {
            var actionSize = 2 + property.FindPropertyRelative(actionSizePropName).arraySize;
            if (property.FindPropertyRelative(actionTypePropName).enumValueIndex == 0)
            {
                actionSize += 1;
            }
            return actionSize * lineHeight;
        }

        /// <summary>
        /// Draws the Vector Actions descriptions for the Brain Parameters
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the BrainParameters
        /// to make the custom GUI for.</param>
        private void DrawVectorActionDescriptions(Rect position, SerializedProperty property)
        {
            var bpVectorActionType = property.FindPropertyRelative(actionTypePropName);
            var vecActionSize = property.FindPropertyRelative(actionSizePropName);
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
                property.FindPropertyRelative(actionDescriptionPropName);
            vecActionDescriptions.arraySize = numberOfDescriptions;
            if (bpVectorActionType.enumValueIndex == 1)
            {
                //Continuous case :
                EditorGUI.PropertyField(
                    position,
                    vecActionDescriptions,
                    new GUIContent(
                        "Action Descriptions",
                        "A list of strings used to name the available actions " +
                        "for the Brain."), true);
                position.y += lineHeight;
            }
            else
            {
                // Discrete case :
                EditorGUI.PropertyField(
                    position,
                    vecActionDescriptions,
                    new GUIContent("Branch Descriptions",
                        "A list of strings used to name the available branches " +
                        "for the Brain."), true);
                position.y += lineHeight;
            }
        }
        /// <summary>
        /// The Height required to draw the Action Descriptions
        /// </summary>
        /// <returns>The height of the drawer of the Action Descriptions </returns>
        private float GetHeightDrawVectorActionDescriptions(SerializedProperty property)
        {
            var descriptionSize = 1;
            if (property.FindPropertyRelative(actionDescriptionPropName).isExpanded)
            {
                var descriptions = property.FindPropertyRelative(actionDescriptionPropName);
                descriptionSize += descriptions.arraySize + 1;
            }
            return descriptionSize * lineHeight;
        }
    }
}
