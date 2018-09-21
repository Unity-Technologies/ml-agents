using UnityEngine;
using UnityEditor;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
    
    [CustomEditor(typeof(LearningBrain))]
    public class LearningBrainEditor : Editor
    {

        public override void OnInspectorGUI()
        {
            LearningBrain brain = (LearningBrain) target;
            
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            
            EditorGUILayout.HelpBox("This is not implemented yet.", MessageType.Error);
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("graphModel"), true);
            serializedBrain.ApplyModifiedProperties();
            
        }
    }
}
