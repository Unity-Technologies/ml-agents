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
        /// <param name="src">The BrainParameters that will be copied</param>
        /// <param name="dst">The BrainParameters that will be overwritten</param>
        private static void DeepCopyBrainParameters(
            BrainParameters src, ref BrainParameters dst)
        {
            dst = new BrainParameters()
            {
                vectorObservationSize = src.vectorObservationSize,
                numStackedVectorObservations = src.numStackedVectorObservations,
                vectorActionSize = (int[]) src.vectorActionSize.Clone(),
                cameraResolutions = (resolution[])src.cameraResolutions.Clone(),
                vectorActionDescriptions = (string[])src.vectorActionDescriptions.Clone(),
                vectorActionSpaceType = src.vectorActionSpaceType
            };
        }
        
        public override void OnInspectorGUI()
        {
            var brain = (Brain) target;
            var brainToCopy = EditorGUILayout.ObjectField(
                "Copy Brain Parameters from : ", null, typeof(Brain), false) as Brain;
            if (brainToCopy != null)
            {
                DeepCopyBrainParameters(brainToCopy.brainParameters, ref brain.brainParameters);
            }
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            
            // Draws a horizontal thick line
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        }
    }
}