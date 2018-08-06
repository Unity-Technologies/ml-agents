using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
    [CustomEditor(typeof(Brain))]
    public class BrainEditor : Editor
    {
        [SerializeField] bool _Foldout = true;

        public override void OnInspectorGUI()
        {
            Brain myBrain = (Brain) target;
            SerializedObject serializedBrain = serializedObject;

            if (myBrain.transform.parent == null)
            {
                EditorGUILayout.HelpBox(
                    "A Brain GameObject must be a child of an Academy GameObject!",
                    MessageType.Error);
            }
            else if (myBrain.transform.parent.GetComponent<Academy>() == null)
            {
                EditorGUILayout.HelpBox(
                    "The Parent of a Brain must have an Academy Component attached to it!",
                    MessageType.Error);
            }

            serializedBrain.Update();


            _Foldout = EditorGUILayout.Foldout(_Foldout, "Brain Parameters");
            int indentLevel = EditorGUI.indentLevel;
            if (_Foldout)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.LabelField("Vector Observation");
                EditorGUI.indentLevel++;

                SerializedProperty bpVectorObsSize =
                    serializedBrain.FindProperty("brainParameters.vectorObservationSize");
                EditorGUILayout.PropertyField(bpVectorObsSize, new GUIContent("Space Size",
                    "Length of state " +
                    "vector for brain (In Continuous state space)." +
                    "Or number of possible values (in Discrete state space)."));


                SerializedProperty bpNumStackedVectorObs =
                    serializedBrain.FindProperty("brainParameters.numStackedVectorObservations");
                EditorGUILayout.PropertyField(bpNumStackedVectorObs, new GUIContent(
                    "Stacked Vectors", "Number of states that" +
                                       " will be stacked before beeing fed to the neural network."));

                EditorGUI.indentLevel--;
                SerializedProperty bpCamResol =
                    serializedBrain.FindProperty("brainParameters.cameraResolutions");
                EditorGUILayout.PropertyField(bpCamResol, new GUIContent("Visual Observation",
                    "Describes height, " +
                    "width, and whether to greyscale visual observations for the Brain."), true);

                EditorGUILayout.LabelField("Vector Action");
                EditorGUI.indentLevel++;

                SerializedProperty bpVectorActionType =
                    serializedBrain.FindProperty("brainParameters.vectorActionSpaceType");
                EditorGUILayout.PropertyField(bpVectorActionType, new GUIContent("Space Type",
                    "Corresponds to whether state" +
                    " vector contains a single integer (Discrete) " +
                    "or a series of real-valued floats (Continuous)."));
                if (bpVectorActionType.enumValueIndex == 1)
                {
                    //Continuous case :
                    SerializedProperty bpVectorActionSize =
                        serializedBrain.FindProperty("brainParameters.vectorActionSize");
                    bpVectorActionSize.arraySize = 1;
                    SerializedProperty continuousActionSize =
                        bpVectorActionSize.GetArrayElementAtIndex(0);
                    EditorGUILayout.PropertyField(continuousActionSize, new GUIContent(
                        "Space Size", "Length of continuous action vector."));
                    
                }
                else
                {
                    // Discrete case :
                    SerializedProperty bpVectorActionSize =
                        serializedBrain.FindProperty("brainParameters.vectorActionSize");
                    bpVectorActionSize.arraySize = EditorGUILayout.IntField(
                        "Branches Size", bpVectorActionSize.arraySize);
                    EditorGUI.indentLevel++;
                    for (int branchIndex = 0;
                        branchIndex < bpVectorActionSize.arraySize;
                        branchIndex++)
                    {
                        SerializedProperty branchActionSize =
                            bpVectorActionSize.GetArrayElementAtIndex(branchIndex);
                        EditorGUILayout.PropertyField(branchActionSize, new GUIContent(
                            "Branch " + branchIndex+" Size", 
                            "Number of possible actions for the branch number " + branchIndex+"."));
                    }
                    EditorGUI.indentLevel--;

                }

                try
                {
                    BrainParameters parameters = myBrain.brainParameters;
                    int numberOfDescriptions = 0;
                    if (parameters.vectorActionSpaceType == SpaceType.continuous)
                        numberOfDescriptions = parameters.vectorActionSize[0];
                    else
                        numberOfDescriptions = parameters.vectorActionSize.Length;
                    if (parameters.vectorActionDescriptions == null ||
                        parameters.vectorActionDescriptions.Length != numberOfDescriptions)
                        parameters.vectorActionDescriptions = new string[numberOfDescriptions];
                }
                catch
                {

                }

                if (bpVectorActionType.enumValueIndex == 1)
                {
                    //Continuous case :
                    SerializedProperty bpVectorActionDescription =
                        serializedBrain.FindProperty("brainParameters.vectorActionDescriptions");
                    EditorGUILayout.PropertyField(bpVectorActionDescription, new GUIContent(
                        "Action Descriptions", "A list of strings used to name" +
                                               " the available actions for the Brain."), true);
                }
                else
                {
                    // Discrete case :
                    SerializedProperty bpVectorActionDescription =
                        serializedBrain.FindProperty("brainParameters.vectorActionDescriptions");
                    EditorGUILayout.PropertyField(bpVectorActionDescription, new GUIContent(
                        "Branch Descriptions", "A list of strings used to name" +
                                               " the available branches for the Brain."), true);
                }
            }

            EditorGUI.indentLevel = indentLevel;
            SerializedProperty bt = serializedBrain.FindProperty("brainType");
            EditorGUILayout.PropertyField(bt);

            if (bt.enumValueIndex < 0)
            {
                bt.enumValueIndex = (int) BrainType.Player;
            }

            serializedBrain.ApplyModifiedProperties();

            myBrain.UpdateCoreBrains();
            myBrain.coreBrain.OnInspector();

#if !NET_4_6 && ENABLE_TENSORFLOW
        EditorGUILayout.HelpBox ("You cannot have ENABLE_TENSORFLOW without NET_4_6", MessageType.Error);
        #endif
        }
    }
}
