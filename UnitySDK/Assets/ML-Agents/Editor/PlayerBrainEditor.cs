using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the PlayerBrain class. Defines the default Inspector view for a
    /// PlayerBrain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains. Also exposes the key mappings for either continuous or discrete control
    /// depending on the Vector Action Space Type of the Brain Parameter. These mappings are the
    /// ones that will be used by the PlayerBrain.
    /// </summary>
    [CustomEditor(typeof(PlayerBrain))]
    public class PlayerBrainEditor : BrainEditor
    {
        private const string KeyContinuousPropName = "keyContinuousPlayerActions";
        private const string KeyDiscretePropName = "discretePlayerActions";
        private const string AxisContinuousPropName = "axisContinuousPlayerActions";
        
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Player Brain", EditorStyles.boldLabel);
            var brain = (PlayerBrain) target;
            var serializedBrain = serializedObject;
            base.OnInspectorGUI();
            
            serializedBrain.Update();
            if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                GUILayout.Label("Edit the continuous inputs for your actions", EditorStyles.boldLabel);
                var keyActionsProp = serializedBrain.FindProperty(KeyContinuousPropName);
                var axisActionsProp = serializedBrain.FindProperty(AxisContinuousPropName);
                
                EditorGUILayout.PropertyField(keyActionsProp , true);
                EditorGUILayout.PropertyField(axisActionsProp, true);
                
                var keyContinuous = brain.keyContinuousPlayerActions;
                var axisContinuous = brain.axisContinuousPlayerActions;
                if (keyContinuous == null)
                {
                    keyContinuous = new PlayerBrain.KeyContinuousPlayerAction[0];
                }
                if (axisContinuous == null)
                {
                    axisContinuous = new PlayerBrain.AxisContinuousPlayerAction[0];
                }
                foreach (var action in keyContinuous)
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
                foreach (var action in axisContinuous)
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
                GUILayout.Label("Edit the discrete inputs for your actions", 
                    EditorStyles.boldLabel);
                var dhas = serializedBrain.FindProperty(KeyDiscretePropName);
                serializedBrain.Update();
                EditorGUILayout.PropertyField(dhas, true);
            }
            serializedBrain.ApplyModifiedProperties();
        }
    }
}