//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Helper to update poses when using native OpenVR integration.
//
//=============================================================================

using UnityEngine;

[ExecuteInEditMode]
public class SteamVR_UpdatePoses : MonoBehaviour
{
	void Awake()
	{
		Debug.Log("SteamVR_UpdatePoses has been deprecated - REMOVING");
		DestroyImmediate(this);
	}
}

