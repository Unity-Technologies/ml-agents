using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
[CustomEditor (typeof(Brain))]
public class BrainEditor : Editor
{
	public override void OnInspectorGUI ()
	{
		Brain myBrain = (Brain)target;
		SerializedObject serializedBrain = serializedObject;

		if (myBrain.transform.parent == null) {
			EditorGUILayout.HelpBox ("A Brain GameObject must be a child of an Academy GameObject!", MessageType.Error);
		} else if (myBrain.transform.parent.GetComponent<Academy> () == null) {
			EditorGUILayout.HelpBox ("The Parent of a Brain must have an Academy Component attached to it!", MessageType.Error);
		}

		BrainParameters parameters = myBrain.brainParameters;
		if (parameters.actionDescriptions == null || parameters.actionDescriptions.Length != parameters.actionSize)
			parameters.actionDescriptions = new string[parameters.actionSize];
		
		serializedBrain.Update();
		
		SerializedProperty bp = serializedBrain.FindProperty ("brainParameters");
		EditorGUILayout.PropertyField(bp, true);

		SerializedProperty bt = serializedBrain.FindProperty("brainType");
		EditorGUILayout.PropertyField(bt);

		if (bt.enumValueIndex < 0) {
			bt.enumValueIndex = (int)BrainType.Player;
		}

		serializedBrain.ApplyModifiedProperties();

		myBrain.UpdateCoreBrains ();
		myBrain.coreBrain.OnInspector ();

		#if !NET_4_6 && ENABLE_TENSORFLOW
		EditorGUILayout.HelpBox ("You cannot have ENABLE_TENSORFLOW without NET_4_6", MessageType.Error);
		#endif
	}
}
