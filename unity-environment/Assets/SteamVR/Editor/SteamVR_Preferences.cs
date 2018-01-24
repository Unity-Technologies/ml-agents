//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Preferences pane for how SteamVR plugin behaves.
//
//=============================================================================

using UnityEngine;
using UnityEditor;

public class SteamVR_Preferences
{
	/// <summary>
	/// Should SteamVR automatically enable VR when opening Unity or pressing play.
	/// </summary>
	public static bool AutoEnableVR
	{
		get
		{
			return EditorPrefs.GetBool("SteamVR_AutoEnableVR", true);
		}
		set
		{
			EditorPrefs.SetBool("SteamVR_AutoEnableVR", value);
		}
	}

	[PreferenceItem("SteamVR")]
	static void PreferencesGUI()
	{
		EditorGUILayout.BeginVertical();
		EditorGUILayout.Space();

		// Automatically Enable VR
		{
			string title = "Automatically Enable VR";
			string tooltip = "Should SteamVR automatically enable VR on launch and play?";
			AutoEnableVR = EditorGUILayout.Toggle(new GUIContent(title, tooltip), AutoEnableVR);
			string helpMessage = "To enable VR manually:\n";
			helpMessage += "- go to Edit -> Project Settings -> Player,\n";
			helpMessage += "- tick 'Virtual Reality Supported',\n";
			helpMessage += "- make sure OpenVR is in the 'Virtual Reality SDKs' list.";
			EditorGUILayout.HelpBox(helpMessage, MessageType.Info);
		}

		EditorGUILayout.EndVertical();
	}
}

