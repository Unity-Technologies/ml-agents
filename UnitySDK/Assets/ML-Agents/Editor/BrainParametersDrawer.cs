using System.Xml;
using UnityEngine;
using UnityEditor;

namespace MLAgents
{

    [CustomPropertyDrawer(typeof(BrainParameters))]
    public class BrainParametersDrawer : PropertyDrawer
    {
        private const float lineHeight = 17f;
        private static int defaultWidth = 84;
        private static int defaultHeight = 84;
        private static bool defaultGray = false;
        
        
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            if (property.isExpanded)
            {
                var vecObsSize = 4;
                var visObsSize =
                    2 +
                    ((property.FindPropertyRelative("cameraResolutions").arraySize > 0) ? 1 : 0) +
                    property.FindPropertyRelative("cameraResolutions").arraySize;
                var actionSize = 1 + property.FindPropertyRelative("vectorActionSize").arraySize;
                if (property.FindPropertyRelative("vectorActionSpaceType").enumValueIndex == 0)
                {
                    actionSize += 1;
                }
                var descriptionSize = 1 + ((property.FindPropertyRelative(
                                          "vectorActionDescriptions").isExpanded)
                                          ? property.FindPropertyRelative(
                                                "vectorActionDescriptions").arraySize + 1
                                          : 0);


                return lineHeight * (1 + vecObsSize + visObsSize + actionSize + descriptionSize);
            }
            return lineHeight;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {

            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            position.height = lineHeight;
            // Brain Parameters
            property.isExpanded = EditorGUI.Foldout(position, property.isExpanded, label);
            position.y += lineHeight;
            

            if (property.isExpanded)
            {
                EditorGUI.BeginProperty(position, label, property);

                EditorGUI.indentLevel++;
                
                // Vector Observations
                EditorGUI.LabelField(position, "Vector Observation");
                position.y += lineHeight;

                EditorGUI.indentLevel++;
                EditorGUI.PropertyField(position,
                    property.FindPropertyRelative("vectorObservationSize"),
                    new GUIContent("Space Size",
                        "Length of state " +
                        "vector for brain (In Continuous state space)." +
                        "Or number of possible values (in Discrete state space)."));
                position.y += lineHeight;
                
                EditorGUI.PropertyField(position,
                    property.FindPropertyRelative("numStackedVectorObservations"),
                    new GUIContent("Stacked Vectors",
                        "Number of states that will be stacked before " +
                        "beeing fed to the neural network."));
                position.y += lineHeight;
                EditorGUI.indentLevel--;                
                
                // Visual Observations             
                EditorGUI.LabelField(position, "Visual Observations");
                position.y += lineHeight;
                var width = position.width / 4;
                var resolutions = property.FindPropertyRelative("cameraResolutions");
                var addButtonRect = position;
                addButtonRect.x += width * 0.5f;
                addButtonRect.width = width * 1.5f;
                if (resolutions.arraySize == 0)
                {
                    addButtonRect.width = addButtonRect.width * 2;
                }
                
                // Display the buttons
                if (GUI.Button(addButtonRect, "Add New", EditorStyles.miniButton))
                {
                    resolutions.arraySize += 1;
                    var newRes = resolutions.GetArrayElementAtIndex(resolutions.arraySize - 1);
                    newRes.FindPropertyRelative("width").intValue = defaultWidth;
                    newRes.FindPropertyRelative("height").intValue = defaultHeight;
                    newRes.FindPropertyRelative("blackAndWhite").boolValue = defaultGray;

                }
                if (resolutions.arraySize > 0)
                {
                    var removeButtonRect = position;
                    removeButtonRect.x += addButtonRect.x + addButtonRect.width;
                    removeButtonRect.width = addButtonRect.width;
                    if (GUI.Button(removeButtonRect, "Remove Last", EditorStyles.miniButton))
                    {
                        resolutions.arraySize -= 1;
                    }
                }
                position.y += lineHeight;
                
                
                // Display the labels for the columns
                var indexRect = position;
                indexRect.width = position.width/4;
                var widthRect = indexRect;
                widthRect.x += width;
                var heightRect = widthRect;
                heightRect.x += width;
                var bwRect = heightRect;
                bwRect.x += width;
                EditorGUI.indentLevel++;
                if (resolutions.arraySize > 0)
                {
                    EditorGUI.LabelField(indexRect, "Index");
                    EditorGUI.LabelField(widthRect, "Width");
                    EditorGUI.LabelField(heightRect, "Height");
                    EditorGUI.LabelField(bwRect, "Gray");
                    position.y += lineHeight;
                }

                // Iterate over the resolutions
                for (var i = 0; i < resolutions.arraySize; i++)
                {
                    indexRect = position;
                    indexRect.width = position.width/4;
                    widthRect = indexRect;
                    widthRect.x += width;
                    heightRect = widthRect;
                    heightRect.x += width;
                    bwRect = heightRect;
                    bwRect.x += width;
                    EditorGUI.LabelField(indexRect, "Obs " + i);
                    var res = resolutions.GetArrayElementAtIndex(i);
                    var w = res.FindPropertyRelative("width");
                    w.intValue = EditorGUI.IntField(widthRect, w.intValue);
                    var h = res.FindPropertyRelative("height");
                    h.intValue = EditorGUI.IntField(heightRect, h.intValue);
                    var bw = res.FindPropertyRelative("blackAndWhite");
                    bw.boolValue = EditorGUI.Toggle(bwRect, bw.boolValue);

                    position.y += lineHeight;
                }
                EditorGUI.indentLevel--;
                
                
                
                // Vector Action
                EditorGUI.LabelField(position, "Vector Action"); 
                position.y += lineHeight;
                EditorGUI.indentLevel++;
                var bpVectorActionType = property.FindPropertyRelative("vectorActionSpaceType");
                EditorGUI.PropertyField(
                    position,
                    bpVectorActionType,
                    new GUIContent("Space Type",
                        "Corresponds to whether state vector contains " +
                        "a single integer (Discrete) " +
                        "or a series of real-valued floats (Continuous)."));
                position.y += lineHeight;
                
                var vectorActionSize =
                    property.FindPropertyRelative("vectorActionSize");
                if (bpVectorActionType.enumValueIndex == 1)
                {
                    //Continuous case :
                    vectorActionSize.arraySize = 1;
                    SerializedProperty continuousActionSize =
                        vectorActionSize.GetArrayElementAtIndex(0);
                    EditorGUI.PropertyField(
                        position,
                        continuousActionSize,
                        new GUIContent("Space Size",
                            "Length of continuous action vector."));
                    position.y += lineHeight;
                }
                else
                {
                    // Discrete case :
                    vectorActionSize.arraySize = EditorGUI.IntField(
                        position,
                        "Branches Size",
                        vectorActionSize.arraySize);
                    position.y += lineHeight;

                    EditorGUI.indentLevel++;
                    for (var branchIndex = 0;
                        branchIndex < vectorActionSize.arraySize;
                        branchIndex++)
                    {
                        SerializedProperty branchActionSize =
                            vectorActionSize.GetArrayElementAtIndex(branchIndex);
                        EditorGUI.PropertyField(
                            position,
                            branchActionSize,
                            new GUIContent(
                                "Branch " + branchIndex + " Size",
                                "Number of possible actions for the " +
                                "branch number " + branchIndex + "."));
                        position.y += lineHeight;
                    }

                    EditorGUI.indentLevel--;
                }

                var numberOfDescriptions = 0;
                if (bpVectorActionType.enumValueIndex == 1)
                {
                    numberOfDescriptions = vectorActionSize.GetArrayElementAtIndex(0).intValue;
                }
                else
                {
                    numberOfDescriptions = vectorActionSize.arraySize;
                }

                EditorGUI.indentLevel++;
                var vectorActionDescriptions =
                    property.FindPropertyRelative("vectorActionDescriptions");
                vectorActionDescriptions.arraySize = numberOfDescriptions;
                if (bpVectorActionType.enumValueIndex == 1)
                {
                    //Continuous case :
                    EditorGUI.PropertyField(
                        position,
                        vectorActionDescriptions,
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
                        vectorActionDescriptions,
                        new GUIContent("Branch Descriptions",
                            "A list of strings used to name the available branches " +
                            "for the Brain."), true);
                    position.y += lineHeight;
                }
                EditorGUI.EndProperty();
            }
            EditorGUI.indentLevel = indent;
        }
    }
}
