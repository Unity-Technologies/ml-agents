//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Example menu using OnGUI with SteamVR_Camera's overlay support
//
//=============================================================================

using UnityEngine;
using Valve.VR;

public class SteamVR_Menu : MonoBehaviour
{
	public Texture cursor, background, logo;
	public float logoHeight, menuOffset;

	public Vector2 scaleLimits = new Vector2(0.1f, 5.0f);
	public float scaleRate = 0.5f;

	SteamVR_Overlay overlay;
	Camera overlayCam;
	Vector4 uvOffset;
	float distance;

	public RenderTexture texture { get { return overlay ? overlay.texture as RenderTexture : null; } }
	public float scale { get; private set; }

	string scaleLimitX, scaleLimitY, scaleRateText;

	CursorLockMode savedCursorLockState;
	bool savedCursorVisible;

	void Awake()
	{
		scaleLimitX = string.Format("{0:N1}", scaleLimits.x);
		scaleLimitY = string.Format("{0:N1}", scaleLimits.y);
		scaleRateText = string.Format("{0:N1}", scaleRate);

		var overlay = SteamVR_Overlay.instance;
		if (overlay != null)
		{
			uvOffset = overlay.uvOffset;
			distance = overlay.distance;
		}
	}

	void OnGUI()
	{
		if (overlay == null)
			return;

		var texture = overlay.texture as RenderTexture;

		var prevActive = RenderTexture.active;
		RenderTexture.active = texture;

		if (Event.current.type == EventType.Repaint)
			GL.Clear(false, true, Color.clear);

		var area = new Rect(0, 0, texture.width, texture.height);

		// Account for screen smaller than texture (since mouse position gets clamped)
		if (Screen.width < texture.width)
		{
			area.width = Screen.width;
			overlay.uvOffset.x = -(float)(texture.width - Screen.width) / (2 * texture.width);
		}
		if (Screen.height < texture.height)
		{
			area.height = Screen.height;
			overlay.uvOffset.y = (float)(texture.height - Screen.height) / (2 * texture.height);
		}

		GUILayout.BeginArea(area);

		if (background != null)
		{
			GUI.DrawTexture(new Rect(
				(area.width - background.width) / 2,
				(area.height - background.height) / 2,
				background.width, background.height), background);
		}

		GUILayout.BeginHorizontal();
		GUILayout.FlexibleSpace();
		GUILayout.BeginVertical();

		if (logo != null)
		{
			GUILayout.Space(area.height / 2 - logoHeight);
			GUILayout.Box(logo);
		}

		GUILayout.Space(menuOffset);

		bool bHideMenu = GUILayout.Button("[Esc] - Close menu");

		GUILayout.BeginHorizontal();
		GUILayout.Label(string.Format("Scale: {0:N4}", scale));
		{
			var result = GUILayout.HorizontalSlider(scale, scaleLimits.x, scaleLimits.y);
			if (result != scale)
			{
				SetScale(result);
			}
		}
		GUILayout.EndHorizontal();

		GUILayout.BeginHorizontal();
		GUILayout.Label(string.Format("Scale limits:"));
		{
			var result = GUILayout.TextField(scaleLimitX);
			if (result != scaleLimitX)
			{
				if (float.TryParse(result, out scaleLimits.x))
					scaleLimitX = result;
			}
		}
		{
			var result = GUILayout.TextField(scaleLimitY);
			if (result != scaleLimitY)
			{
				if (float.TryParse(result, out scaleLimits.y))
					scaleLimitY = result;
			}
		}
		GUILayout.EndHorizontal();

		GUILayout.BeginHorizontal();
		GUILayout.Label(string.Format("Scale rate:"));
		{
			var result = GUILayout.TextField(scaleRateText);
			if (result != scaleRateText)
			{
				if (float.TryParse(result, out scaleRate))
					scaleRateText = result;
			}
		}
		GUILayout.EndHorizontal();

		if (SteamVR.active)
		{
			var vr = SteamVR.instance;

			GUILayout.BeginHorizontal();
			{
				var t = SteamVR_Camera.sceneResolutionScale;
				int w = (int)(vr.sceneWidth * t);
				int h = (int)(vr.sceneHeight * t);
				int pct = (int)(100.0f * t);
				GUILayout.Label(string.Format("Scene quality: {0}x{1} ({2}%)", w, h, pct));
				var result = Mathf.RoundToInt(GUILayout.HorizontalSlider(pct, 50, 200));
				if (result != pct)
				{
					SteamVR_Camera.sceneResolutionScale = (float)result / 100.0f;
				}
			}
			GUILayout.EndHorizontal();
		}

		overlay.highquality = GUILayout.Toggle(overlay.highquality, "High quality");

		if (overlay.highquality)
		{
			overlay.curved = GUILayout.Toggle(overlay.curved, "Curved overlay");
			overlay.antialias = GUILayout.Toggle(overlay.antialias, "Overlay RGSS(2x2)");
		}
		else
		{
			overlay.curved = false;
			overlay.antialias = false;
		}

		var tracker = SteamVR_Render.Top();
		if (tracker != null)
		{
			tracker.wireframe = GUILayout.Toggle(tracker.wireframe, "Wireframe");

			var render = SteamVR_Render.instance;
			if (render.trackingSpace == ETrackingUniverseOrigin.TrackingUniverseSeated)
			{
				if (GUILayout.Button("Switch to Standing"))
					render.trackingSpace = ETrackingUniverseOrigin.TrackingUniverseStanding;
				if (GUILayout.Button("Center View"))
				{
					var system = OpenVR.System;
					if (system != null)
						system.ResetSeatedZeroPose();
				}
			}
			else
			{
				if (GUILayout.Button("Switch to Seated"))
					render.trackingSpace = ETrackingUniverseOrigin.TrackingUniverseSeated;
			}
		}

#if !UNITY_EDITOR
		if (GUILayout.Button("Exit"))
			Application.Quit();
#endif
		GUILayout.Space(menuOffset);

		var env = System.Environment.GetEnvironmentVariable("VR_OVERRIDE");
		if (env != null)
		{
			GUILayout.Label("VR_OVERRIDE=" + env);
		}

		GUILayout.Label("Graphics device: " + SystemInfo.graphicsDeviceVersion);

		GUILayout.EndVertical();
		GUILayout.FlexibleSpace();
		GUILayout.EndHorizontal();

		GUILayout.EndArea();

		if (cursor != null)
		{
			float x = Input.mousePosition.x, y = Screen.height - Input.mousePosition.y;
			float w = cursor.width, h = cursor.height;
			GUI.DrawTexture(new Rect(x, y, w, h), cursor);
		}

		RenderTexture.active = prevActive;

		if (bHideMenu)
			HideMenu();
	}

	public void ShowMenu()
	{
		var overlay = SteamVR_Overlay.instance;
		if (overlay == null)
			return;

		var texture = overlay.texture as RenderTexture;
		if (texture == null)
		{
			Debug.LogError("Menu requires overlay texture to be a render texture.");
			return;
		}

		SaveCursorState();

		Cursor.visible = true;
		Cursor.lockState = CursorLockMode.None;

		this.overlay = overlay;
		uvOffset = overlay.uvOffset;
		distance = overlay.distance;

		// If an existing camera is rendering into the overlay texture, we need
		// to temporarily disable it to keep it from clearing the texture on us.
		var cameras = Object.FindObjectsOfType(typeof(Camera)) as Camera[];
		foreach (var cam in cameras)
		{
			if (cam.enabled && cam.targetTexture == texture)
			{
				overlayCam = cam;
				overlayCam.enabled = false;
				break;
			}
		}

		var tracker = SteamVR_Render.Top();
		if (tracker != null)
			scale = tracker.origin.localScale.x;
	}

	public void HideMenu()
	{
		RestoreCursorState();

		if (overlayCam != null)
			overlayCam.enabled = true;

		if (overlay != null)
		{
			overlay.uvOffset = uvOffset;
			overlay.distance = distance;
			overlay = null;
		}
	}

	void Update()
	{
		if (Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyDown(KeyCode.Joystick1Button7))
		{
			if (overlay == null)
			{
				ShowMenu();
			}
			else
			{
				HideMenu();
			}
		}
		else if (Input.GetKeyDown(KeyCode.Home))
		{
			SetScale(1.0f);
		}
		else if (Input.GetKey(KeyCode.PageUp))
		{
			SetScale(Mathf.Clamp(scale + scaleRate * Time.deltaTime, scaleLimits.x, scaleLimits.y));
		}
		else if (Input.GetKey(KeyCode.PageDown))
		{
			SetScale(Mathf.Clamp(scale - scaleRate * Time.deltaTime, scaleLimits.x, scaleLimits.y));
		}
	}

	void SetScale(float scale)
	{
		this.scale = scale;

		var tracker = SteamVR_Render.Top();
		if (tracker != null)
			tracker.origin.localScale = new Vector3(scale, scale, scale);
	}

	void SaveCursorState()
	{
		savedCursorVisible = Cursor.visible;
		savedCursorLockState = Cursor.lockState;
	}

	void RestoreCursorState()
	{
		Cursor.visible = savedCursorVisible;
		Cursor.lockState = savedCursorLockState;
	}
}

