using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;

namespace MLAgents
{
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the 
 inspector accordingly.
*/
    [CustomEditor(typeof(Brain))]
    public class BrainEditor : Editor
    {

        private void CopyBrainParametersFrom(Brain originalBrain)
        {
            var brainToCopy = EditorGUILayout.ObjectField("Copy Brain Parameters from : ", null,
                typeof(Brain), false) as Brain;
            if (brainToCopy!=null)
            {
                var newParams = brainToCopy.brainParameters;
                originalBrain.brainParameters = new BrainParameters()
                {
                    vectorObservationSize = newParams.vectorObservationSize,
                    numStackedVectorObservations = newParams.numStackedVectorObservations,
                    vectorActionSize = (int[]) newParams.vectorActionSize.Clone(),
                    cameraResolutions = (resolution[])newParams.cameraResolutions.Clone(),
                    vectorActionDescriptions = (string[])newParams.vectorActionDescriptions.Clone(),
                    vectorActionSpaceType = newParams.vectorActionSpaceType
                };
            }
        }
        
        public override void OnInspectorGUI()
        {
            var brain = (Brain) target;
            CopyBrainParametersFrom(brain);
            var serializedBrain = serializedObject;
            serializedBrain.Update(); 
            EditorGUILayout.PropertyField(serializedBrain.FindProperty("brainParameters"), true);
            serializedBrain.ApplyModifiedProperties();
            
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
        }
    }
}