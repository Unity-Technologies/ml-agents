using UnityEngine;
using UnityEditor;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
    
    [CustomEditor(typeof(HeuristicBrain))]
    public class ScriptableBrainEditor : Editor
    {

        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Heuristic Brain", EditorStyles.boldLabel);
            HeuristicBrain brain = (HeuristicBrain) target;
            
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            
            brain.decisionScript = EditorGUILayout.ObjectField(
                "Decision Script", brain.decisionScript, typeof(MonoScript), true) as MonoScript;

            /*If the monoscript is not a decision then do not validate it*/

            if (brain.decisionScript != null)
            {
                var decisionInstance = (CreateInstance(brain.decisionScript.name) as Decision);
                if (decisionInstance == null)
                {
                    brain.decisionScript = null;
                    Debug.LogError(
                        "Instance of " + brain.decisionScript.name + " couldn't be created. " +
                        "The the script class needs to derive from ScriptableObject.");
                }
            }

            if (brain.decisionScript == null)
            {
                EditorGUILayout.HelpBox("You need to add a 'Decision' component to this Object",
                    MessageType.Error);
            }
        }
    }
}
