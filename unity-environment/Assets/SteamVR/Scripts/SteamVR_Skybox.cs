//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Sets cubemap to use in the compositor.
//
//=============================================================================

using UnityEngine;
using Valve.VR;

public class SteamVR_Skybox : MonoBehaviour
{
	// Note: Unity's Left and Right Skybox shader variables are switched.
	public Texture front, back, left, right, top, bottom;

	public enum CellSize
	{
		x1024, x64, x32, x16, x8
	}
	public CellSize StereoCellSize = CellSize.x32;

	public float StereoIpdMm = 64.0f;

	public void SetTextureByIndex(int i, Texture t)
	{
		switch (i)
		{
			case 0:
				front = t;
				break;
			case 1:
				back = t;
				break;
			case 2:
				left = t;
				break;
			case 3:
				right = t;
				break;
			case 4:
				top = t;
				break;
			case 5:
				bottom = t;
				break;
		}
	}

	public Texture GetTextureByIndex(int i)
	{
		switch (i)
		{
			case 0:
				return front;
			case 1:
				return back;
			case 2:
				return left;
			case 3:
				return right;
			case 4:
				return top;
			case 5:
				return bottom;
		}
		return null;
	}

	static public void SetOverride(
		Texture front = null,
		Texture back = null,
		Texture left = null,
		Texture right = null,
		Texture top = null,
		Texture bottom = null )
	{
		var compositor = OpenVR.Compositor;
		if (compositor != null)
		{
			var handles = new Texture[] { front, back, left, right, top, bottom };
			var textures = new Texture_t[6];
			for (int i = 0; i < 6; i++)
			{
				textures[i].handle = (handles[i] != null) ? handles[i].GetNativeTexturePtr() : System.IntPtr.Zero;
				textures[i].eType = SteamVR.instance.textureType;
				textures[i].eColorSpace = EColorSpace.Auto;
			}
			var error = compositor.SetSkyboxOverride(textures);
			if (error != EVRCompositorError.None)
			{
				Debug.LogError("Failed to set skybox override with error: " + error);
				if (error == EVRCompositorError.TextureIsOnWrongDevice)
					Debug.Log("Set your graphics driver to use the same video card as the headset is plugged into for Unity.");
				else if (error == EVRCompositorError.TextureUsesUnsupportedFormat)
					Debug.Log("Ensure skybox textures are not compressed and have no mipmaps.");
			}
		}
	}

	static public void ClearOverride()
	{
		var compositor = OpenVR.Compositor;
		if (compositor != null)
			compositor.ClearSkyboxOverride();
	}

	void OnEnable()
	{
		SetOverride(front, back, left, right, top, bottom);
	}

	void OnDisable()
	{
		ClearOverride();
	}
}

