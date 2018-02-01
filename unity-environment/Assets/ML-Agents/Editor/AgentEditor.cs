using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEditor;
using System.Linq;
/*
 This code is meant to modify the behavior of the inspector on Brain Components.
 Depending on the type of brain that is used, the available fields will be modified in the inspector accordingly.
*/
[CustomEditor (typeof(Agent), true)]
[CanEditMultipleObjects]
public class AgentEditor : Editor
{
    
	public override void OnInspectorGUI ()
	{
        Agent myAgent = (Agent)target;
        //EditorGUILayout.HelpBox("Erroblahr", MessageType.Error);
        myAgent.done = EditorGUILayout.Toggle("done", myAgent.done);
		base.OnInspectorGUI();

	}
}
