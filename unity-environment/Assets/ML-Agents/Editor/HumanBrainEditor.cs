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
                GUILayout.Label("Edit the continuous inputs for your actions",
                    EditorStyles.boldLabel);
                var chas = serializedBrain.FindProperty("continuousPlayerActions");
                serializedBrain.Update();
                EditorGUILayout.PropertyField(chas, true);
                serializedBrain.ApplyModifiedProperties();
                if (brain.continuousPlayerActions == null)
                {
                    brain.continuousPlayerActions = new HumanBrain.ContinuousPlayerAction[0];
                }

                foreach (HumanBrain.ContinuousPlayerAction cha in brain.continuousPlayerActions)
                {
                    if (cha.index >= brain.brainParameters.vectorActionSize)
                    {
                        EditorGUILayout.HelpBox(string.Format(
                            "Key {0} is assigned to index {1} but the action size is only of size {2}"
                            , cha.key.ToString(), cha.index.ToString(),
                            brain.brainParameters.vectorActionSize.ToString()), MessageType.Error);
                    }
                }

            }
            else
            {
                GUILayout.Label("Edit the discrete inputs for your actions",
                    EditorStyles.boldLabel);
                brain.defaultAction = EditorGUILayout.IntField("Default Action", brain.defaultAction);
                var dhas = serializedBrain.FindProperty("discretePlayerActions");
                serializedBrain.Update();
                EditorGUILayout.PropertyField(dhas, true);
                serializedBrain.ApplyModifiedProperties();
            }
        }
    }
}
