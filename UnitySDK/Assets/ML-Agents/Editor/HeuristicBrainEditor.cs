using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the Heuristic Brain class. Defines the default Inspector view for a
    /// HeuristicBrain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains. Provides a drag box for a Decision Monoscript that will be used by
    /// the Heuristic Brain.
    /// </summary>
    [CustomEditor(typeof(HeuristicBrain))]
    public class ScriptableBrainEditor : BrainEditor
    {

        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Heuristic Brain", EditorStyles.boldLabel);
            var brain = (HeuristicBrain) target;
            base.OnInspectorGUI();
            
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
