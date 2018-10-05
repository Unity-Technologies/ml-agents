using UnityEngine;
using UnityEditor;
using UnityEngine.MachineLearning.InferenceEngine;

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
        private const string ModelPropName = "model";
        private const string DevicePropName = "inferenceDevice";
        private const float TimeBetweenModelReloads = 2f;
        private float _timeSinceModelReload;
        private bool _requireReload;
        
        /// <summary>
        /// Is called once when the user opens the Inspector for the LearningBrain
        /// </summary>
        public void OnEnable()
        {
            _requireReload = true;
            EditorApplication.update += Update;
        }
        /// <summary>
        /// Is called once when the user leaves the Inspector for the LearningBrain
        /// </summary>
        public void OnDisable()
        {
            EditorApplication.update -= Update;
        }
        
        public override void OnInspectorGUI()
        {
            EditorGUILayout.LabelField("Learning Brain", EditorStyles.boldLabel);
            var brain = (LearningBrain) target;
            var serializedBrain = serializedObject;
            EditorGUI.BeginChangeCheck();
            base.OnInspectorGUI();
            serializedBrain.Update(); 
            var tfGraphModel = serializedBrain.FindProperty(ModelPropName);
            EditorGUILayout.ObjectField(tfGraphModel);
            var deviceType = serializedBrain.FindProperty(DevicePropName);
            EditorGUILayout.PropertyField(deviceType);
            serializedBrain.ApplyModifiedProperties();

            if (EditorGUI.EndChangeCheck())
            {
                _requireReload = true;
            }
            if (_requireReload && _timeSinceModelReload > TimeBetweenModelReloads)
            {
                brain.GiveModel(brain.model);
                _requireReload = false;
                _timeSinceModelReload = 0;
            }
            foreach (var error in brain.GetModelFailedChecks())
            {
                if (error != null)
                    EditorGUILayout.HelpBox(error, MessageType.Warning);
            }
        }

        /// <summary>
        /// Is called once every EditorApplication update 
        /// </summary>
        private void Update()
        {
            _timeSinceModelReload += Time.deltaTime;
        }
    }
}
