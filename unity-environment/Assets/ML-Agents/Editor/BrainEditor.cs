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
		SerializedObject serializedBrain = new SerializedObject (target);

		if (myBrain.transform.parent == null) {
			EditorGUILayout.HelpBox ("A Brain GameObject myst be a child of an Academy GameObject!", MessageType.Error);
		} else if (myBrain.transform.parent.GetComponent<Academy> () == null) {
			EditorGUILayout.HelpBox ("The Parent of a Brain must have an Academy Component attached to it!", MessageType.Error);
		} 


		SerializedProperty bp = serializedBrain.FindProperty ("brainParameters");
		if (myBrain.brainParameters.actionDescriptions == null) {
			myBrain.brainParameters.actionDescriptions = new string[myBrain.brainParameters.actionSize];
		}
		if (myBrain.brainParameters.actionSize != myBrain.brainParameters.actionDescriptions.Count()) {
			myBrain.brainParameters.actionDescriptions = new string[myBrain.brainParameters.actionSize];
		}
		serializedBrain.Update ();
		EditorGUILayout.PropertyField (bp, true);
		serializedBrain.ApplyModifiedProperties ();

		myBrain.brainType = (BrainType)EditorGUILayout.EnumPopup ("Type Of Brain ", myBrain.brainType);

		if ((int)myBrain.brainType >= System.Enum.GetValues (typeof(BrainType)).Length) {
			myBrain.brainType = BrainType.Player;
		}

		myBrain.UpdateCoreBrains ();

		myBrain.coreBrain.OnInspector ();

		#if !NET_4_6 && ENABLE_TENSORFLOW
		EditorGUILayout.HelpBox ("You cannot have ENABLE_TENSORFLOW without NET_4_6", MessageType.Error);
		#endif


	}
}
