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
    [CustomEditor(typeof(PlayerBrain))]
    public class PlayerBrainEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Player Brain", EditorStyles.boldLabel);
            PlayerBrain brain = (PlayerBrain) target;
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            serializedBrain.Update();
            if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                GUILayout.Label("Edit the continuous inputs for your actions", EditorStyles.boldLabel);
                var keyActionsProp = serializedBrain.FindProperty("keyContinuousPlayerActions");
                var axisActionsProp = serializedBrain.FindProperty("axisContinuousPlayerActions");
                
                EditorGUILayout.PropertyField(keyActionsProp , true);
                EditorGUILayout.PropertyField(axisActionsProp, true);
                
                PlayerBrain.KeyContinuousPlayerAction[] keyContinuous =
                    brain.keyContinuousPlayerActions;
                PlayerBrain.AxisContinuousPlayerAction[] axisContinuous =
                    brain.axisContinuousPlayerActions;
                if (keyContinuous == null)
                {
                    keyContinuous = new PlayerBrain.KeyContinuousPlayerAction[0];
                }
                if (axisContinuous == null)
                {
                    axisContinuous = new PlayerBrain.AxisContinuousPlayerAction[0];
                }
                foreach (PlayerBrain.KeyContinuousPlayerAction action in keyContinuous)
                {
                    if (action.index >= brain.brainParameters.vectorActionSize[0])
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
                foreach (PlayerBrain.AxisContinuousPlayerAction action in axisContinuous)
                {
                    if (action.index >= brain.brainParameters.vectorActionSize[0])
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
                var dhas = serializedBrain.FindProperty("discretePlayerActions");
                serializedBrain.Update();
                EditorGUILayout.PropertyField(dhas, true);
            }
            serializedBrain.ApplyModifiedProperties();
        }
    }
}