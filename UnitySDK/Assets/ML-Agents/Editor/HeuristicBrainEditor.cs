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
    public class HeuristicBrainEditor : BrainEditor
    {
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Heuristic Brain", EditorStyles.boldLabel);
            var brain = (HeuristicBrain) target;
            base.OnInspectorGUI();
            
            // Expose the Heuristic Brain's Monoscript for decision in a drag and drop box.
            brain.decisionScript = EditorGUILayout.ObjectField(
                "Decision Script", brain.decisionScript, typeof(MonoScript), true) as MonoScript;

            CheckIsDecision(brain);
            // Draw an error box if the Decision is not set.
            if (brain.decisionScript == null)
            {
                EditorGUILayout.HelpBox("You need to add a 'Decision' component to this Object",
                    MessageType.Error);
            }
        }

        /// <summary>
        /// Ensures tht the Monoscript for the decision of the HeuristicBrain is either null or
        /// an implementation of Decision. If the Monoscript is not an implementation of
        /// Decision, it will be set to null.
        /// </summary>
        /// <param name="brain">The HeuristicBrain with the decision script attached</param>
        private static void CheckIsDecision(HeuristicBrain brain)
        {
            if (brain.decisionScript != null)
            {
                var decisionInstance = (CreateInstance(brain.decisionScript.name) as Decision);
                if (decisionInstance == null)
                {
                    Debug.LogError(
                        "Instance of " + brain.decisionScript.name + " couldn't be created. " +
                        "The the script class needs to derive from Decision.");
                    brain.decisionScript = null;
                }
            }
        }
    }
}
