using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the 
 inspector accordingly.
*/
    [CustomEditor(typeof(HumanBrain))]
    public class HumanBrainEditor : BrainEditor
    {
        

        public override void OnInspectorGUI()
        {
            HumanBrain brain = (HumanBrain) target;
            base.OnInspectorGUI();
            
            SerializedObject serializedBrain = serializedObject;
            if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
            GUILayout.Label(
                "Edit the continuous inputs for your actions", 
                EditorStyles.boldLabel);
            var keyActionsProp = serializedBrain.FindProperty("keyContinuousPlayerActions");
            var axisActionsProp = serializedBrain.FindProperty("axisContinuousPlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(keyActionsProp , true);
            EditorGUILayout.PropertyField(axisActionsProp, true);
            serializedBrain.ApplyModifiedProperties();
            if (brain.keyContinuousPlayerActions == null)
            {
                brain.keyContinuousPlayerActions = new HumanBrain.KeyContinuousPlayerAction[0];
            }
            if (brain.axisContinuousPlayerActions == null)
            {
                brain.axisContinuousPlayerActions = new HumanBrain.AxisContinuousPlayerAction[0];
            }
            foreach (var action in brain.keyContinuousPlayerActions)
            {
                if (action.index >= brain.brainParameters.vectorActionSize)
                {
                    EditorGUILayout.HelpBox(
                        string.Format(
                            "Key {0} is assigned to index {1} " +
                            "but the action size is only of size {2}"
                        , action.key.ToString(), action.index.ToString(), 
                            brain.brainParameters.vectorActionSize.ToString()), 
                        MessageType.Error);
                }
            }
            foreach (var action in brain.axisContinuousPlayerActions)
            {
                if (action .index >= brain.brainParameters.vectorActionSize)
                {
                    EditorGUILayout.HelpBox(
                        string.Format(
                            "Axis {0} is assigned to index {1} " +
                            "but the action size is only of size {2}"
                        , action.axis, action.index.ToString(),
                            brain.brainParameters.vectorActionSize.ToString()), 
                        MessageType.Error);
                }
            }
            GUILayout.Label("You can change axis settings from Edit->Project Settings->Input", 
                EditorStyles.helpBox );
        }
        else
        {
            GUILayout.Label("Edit the discrete inputs for your actions", EditorStyles.boldLabel);
            brain.defaultAction = EditorGUILayout.IntField("Default Action", brain.defaultAction);
            var dhas = serializedBrain.FindProperty("discretePlayerActions");
            serializedBrain.Update();
            EditorGUILayout.PropertyField(dhas, true);
            serializedBrain.ApplyModifiedProperties();
        }
        }
    }
}
