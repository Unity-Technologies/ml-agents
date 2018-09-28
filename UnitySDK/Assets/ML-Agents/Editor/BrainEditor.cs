using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;

namespace MLAgents
{
    /// <summary>
    /// CustomEditor for the Brain base class. Defines the default Inspector view for a Brain.
    /// Shows the BrainParameters of the Brain and expose a tool to deep copy BrainParameters
    /// between brains.
    /// </summary>
    [CustomEditor(typeof(Brain))]
    public class BrainEditor : Editor
    {
        /// <summary>
        /// DeepCopy of BrainParameters
        /// </summary>
        /// <param name="source">The BrainParameters that will be copied</param>
        /// <param name="target">The BrainParameters that will be overwritten</param>
        private static void DeepCopyBrainParametersFrom(
            BrainParameters source, ref BrainParameters target)
        {
            target = new BrainParameters()
            {
                vectorObservationSize = source.vectorObservationSize,
                numStackedVectorObservations = source.numStackedVectorObservations,
                vectorActionSize = (int[]) source.vectorActionSize.Clone(),
                cameraResolutions = (resolution[])source.cameraResolutions.Clone(),
                vectorActionDescriptions = (string[])source.vectorActionDescriptions.Clone(),
                vectorActionSpaceType = source.vectorActionSpaceType
            };
        }
        
        public override void OnInspectorGUI()
        {
            var brain = (Brain) target;
            var brainToCopy = EditorGUILayout.ObjectField("Copy Brain Parameters from : ", null,
                typeof(Brain), false) as Brain;
            if (brainToCopy != null)
            {
                DeepCopyBrainParametersFrom(brainToCopy.brainParameters, ref brain.brainParameters);
            }
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        }
    }
}