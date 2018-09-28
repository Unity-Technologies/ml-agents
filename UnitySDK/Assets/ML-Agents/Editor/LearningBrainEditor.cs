using UnityEngine;
using UnityEditor;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the LearningBrain class. Defines the default Inspector view for a
    /// LearningBrain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains. Also exposes a drag box for the Model that will be used by the
    /// LearningBrain. 
    /// </summary>
    [CustomEditor(typeof(LearningBrain))]
    public class LearningBrainEditor : BrainEditor
    {
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Learning Brain", EditorStyles.boldLabel);
            
            var brain = (LearningBrain) target;
            
            var serializedBrain = serializedObject;
            base.OnInspectorGUI();
            
            EditorGUILayout.HelpBox("This is not implemented yet.", MessageType.Error);
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("graphModel"), true);
            serializedBrain.ApplyModifiedProperties();
            
        }
    }
}
