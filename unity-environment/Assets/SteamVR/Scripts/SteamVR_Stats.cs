//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Helper to display various hmd stats via GUIText
//
//=============================================================================

using UnityEngine;
using Valve.VR;

public class SteamVR_Stats : MonoBehaviour
{
	public GUIText text;

	public Color fadeColor = Color.black;
	public float fadeDuration = 1.0f;

	void Awake()
	{
		if (text == null)
		{
			text = GetComponent<GUIText>();
			text.enabled = false;
		}

		if (fadeDuration > 0)
		{
			SteamVR_Fade.Start(fadeColor, 0);
			SteamVR_Fade.Start(Color.clear, fadeDuration);
		}
	}

	double lastUpdate = 0.0f;

	void Update()
	{
		if (text != null)
		{
			if (Input.GetKeyDown(KeyCode.I))
			{
				text.enabled = !text.enabled;
			}

			if (text.enabled)
			{
				var compositor = OpenVR.Compositor;
				if (compositor != null)
				{
					var timing = new Compositor_FrameTiming();
					timing.m_nSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(typeof(Compositor_FrameTiming));
					compositor.GetFrameTiming(ref timing, 0);

					var update = timing.m_flSystemTimeInSeconds;
					if (update > lastUpdate)
					{
						var framerate = (lastUpdate > 0.0f) ? 1.0f / (update - lastUpdate) : 0.0f;
						lastUpdate = update;
						text.text = string.Format("framerate: {0:N0}\ndropped frames: {1}", framerate, (int)timing.m_nNumDroppedFrames);
					}
					else
					{
						lastUpdate = update;
					}
				}
			}
		}
	}
}

