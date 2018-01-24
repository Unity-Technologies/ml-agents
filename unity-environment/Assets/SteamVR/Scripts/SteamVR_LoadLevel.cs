//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Helper for smoothing over transitions between levels.
//
//=============================================================================

using UnityEngine;
using System.Collections;
using Valve.VR;
using System.IO;

public class SteamVR_LoadLevel : MonoBehaviour
{
	private static SteamVR_LoadLevel _active = null;
	public static bool loading { get { return _active != null; } }
	public static float progress
	{
		get { return (_active != null && _active.async != null) ? _active.async.progress : 0.0f; }
	}
	public static Texture progressTexture
	{
		get { return (_active != null) ? _active.renderTexture : null; }
	}

	// Name of level to load.
	public string levelName;

	// Name of internal process to launch (instead of levelName).
	public string internalProcessPath;

	// The command-line args for the internal process to launch.
	public string internalProcessArgs;

	// If true, call LoadLevelAdditiveAsync instead of LoadLevelAsync.
	public bool loadAdditive;

	// Async load causes crashes in some apps.
	public bool loadAsync = true;

	// Optional logo texture.
	public Texture loadingScreen;

	// Optional progress bar textures.
	public Texture progressBarEmpty, progressBarFull;

	// Sizes of overlays.
	public float loadingScreenWidthInMeters = 6.0f;
	public float progressBarWidthInMeters = 3.0f;

	// If specified, the loading screen will be positioned in the player's view this far away.
	public float loadingScreenDistance = 0.0f;

	// Optional overrides for where to display loading screen and progress bar overlays.
	// Otherwise defaults to using this object's transform.
	public Transform loadingScreenTransform, progressBarTransform;

	// Optional skybox override textures.
	public Texture front, back, left, right, top, bottom;

	// Colors to use when dropping to the compositor between levels if no skybox is set.
	public Color backgroundColor = Color.black;

	// If false, the background color above gets applied as the foreground color in the compositor.
	// This does not have any effect when using a skybox instead.
	public bool showGrid = false;

	// Time to fade from current scene to the compositor and back.
	public float fadeOutTime = 0.5f;
	public float fadeInTime = 0.5f;

	// Additional time to wait after finished loading before we start fading the new scene back in.
	// This is to cover up any initial hitching that takes place right at the start of levels.
	// Most scenes should hopefully not require this.
	public float postLoadSettleTime = 0.0f;

	// Time to fade loading screen in and out (also used for progress bar).
	public float loadingScreenFadeInTime = 1.0f;
	public float loadingScreenFadeOutTime = 0.25f;

	float fadeRate = 1.0f;
	float alpha = 0.0f;

	AsyncOperation async; // used to track level load progress
	RenderTexture renderTexture; // used to render progress bar

	ulong loadingScreenOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;
	ulong progressBarOverlayHandle = OpenVR.k_ulOverlayHandleInvalid;

	public bool autoTriggerOnEnable = false;

	void OnEnable()
	{
		if (autoTriggerOnEnable)
			Trigger();
	}

	public void Trigger()
	{
		if (!loading && !string.IsNullOrEmpty(levelName))
			StartCoroutine(LoadLevel());
	}

	// Helper function to quickly and simply load a level from script.
	public static void Begin(string levelName,
		bool showGrid = false, float fadeOutTime = 0.5f,
		float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f)
	{
		var loader = new GameObject("loader").AddComponent<SteamVR_LoadLevel>();
		loader.levelName = levelName;
		loader.showGrid = showGrid;
		loader.fadeOutTime = fadeOutTime;
		loader.backgroundColor = new Color(r, g, b, a);
		loader.Trigger();
	}

	// Updates progress bar.
	void OnGUI()
	{
		if (_active != this)
			return;

		// Optionally create an overlay for our progress bar to use, separate from the loading screen.
		if (progressBarEmpty != null && progressBarFull != null)
		{
			if (progressBarOverlayHandle == OpenVR.k_ulOverlayHandleInvalid)
				progressBarOverlayHandle = GetOverlayHandle("progressBar", progressBarTransform != null ? progressBarTransform : transform, progressBarWidthInMeters);

			if (progressBarOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
			{
				var progress = (async != null) ? async.progress : 0.0f;

				// Use the full bar size for everything.
				var w = progressBarFull.width;
				var h = progressBarFull.height;

				// Create a separate render texture so we can composite the full image on top of the empty one.
				if (renderTexture == null)
				{
					renderTexture = new RenderTexture(w, h, 0);
					renderTexture.Create();
				}

				var prevActive = RenderTexture.active;
				RenderTexture.active = renderTexture;

				if (Event.current.type == EventType.Repaint)
					GL.Clear(false, true, Color.clear);

				GUILayout.BeginArea(new Rect(0, 0, w, h));

				GUI.DrawTexture(new Rect(0, 0, w, h), progressBarEmpty);

				// Reveal the full bar texture based on progress.
				GUI.DrawTextureWithTexCoords(new Rect(0, 0, progress * w, h), progressBarFull, new Rect(0.0f, 0.0f, progress, 1.0f));

				GUILayout.EndArea();

				RenderTexture.active = prevActive;

				// Texture needs to be set every frame after it is updated since SteamVR makes a copy internally to a shared texture.
				var overlay = OpenVR.Overlay;
				if (overlay != null)
				{
					var texture = new Texture_t();
					texture.handle = renderTexture.GetNativeTexturePtr();
					texture.eType = SteamVR.instance.textureType;
					texture.eColorSpace = EColorSpace.Auto;
					overlay.SetOverlayTexture(progressBarOverlayHandle, ref texture);
				}
			}
		}

		#if false
		// Draw loading screen and progress bar to 2d companion window as well.
		if (loadingScreen != null)
		{
			var screenAspect = (float)Screen.width / Screen.height;
			var textureAspect = (float)loadingScreen.width / loadingScreen.height;

			float w, h;
			if (screenAspect < textureAspect)
			{
				// Clamp horizontally
				w = Screen.width * 0.9f;
				h = w / textureAspect;
			}
			else
			{
				// Clamp vertically
				h = Screen.height * 0.9f;
				w = h * textureAspect;
			}

			GUILayout.BeginArea(new Rect(0, 0, Screen.width, Screen.height));

			var x = Screen.width / 2 - w / 2;
			var y = Screen.height / 2 - h / 2;
			GUI.DrawTexture(new Rect(x, y, w, h), loadingScreen);

			GUILayout.EndArea();
		}

		if (renderTexture != null)
		{
			var x = Screen.width / 2 - renderTexture.width / 2;
			var y = Screen.height * 0.9f - renderTexture.height;
			GUI.DrawTexture(new Rect(x, y, renderTexture.width, renderTexture.height), renderTexture);
		}
		#endif
	}

	// Fade our overlays in/out over time.
	void Update()
	{
		if (_active != this)
			return;

		alpha = Mathf.Clamp01(alpha + fadeRate * Time.deltaTime);

		var overlay = OpenVR.Overlay;
		if (overlay != null)
		{
			if (loadingScreenOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
				overlay.SetOverlayAlpha(loadingScreenOverlayHandle, alpha);

			if (progressBarOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
				overlay.SetOverlayAlpha(progressBarOverlayHandle, alpha);
		}
	}

	// Corourtine to handle all the steps across loading boundaries.
	IEnumerator LoadLevel()
	{
		// Optionally rotate loading screen transform around the camera into view.
		// We assume here that the loading screen is already facing toward the origin,
		// and that the progress bar transform (if any) is a child and will follow along.
		if (loadingScreen != null && loadingScreenDistance > 0.0f)
		{
			// Wait until we have tracking.
			var hmd = SteamVR_Controller.Input((int)OpenVR.k_unTrackedDeviceIndex_Hmd);
			while (!hmd.hasTracking)
				yield return null;

			var tloading = hmd.transform;
			tloading.rot = Quaternion.Euler(0.0f, tloading.rot.eulerAngles.y, 0.0f);
			tloading.pos += tloading.rot * new Vector3(0.0f, 0.0f, loadingScreenDistance);

			var t = loadingScreenTransform != null ? loadingScreenTransform : transform;
			t.position = tloading.pos;
			t.rotation = tloading.rot;
		}

		_active = this;

		SteamVR_Events.Loading.Send(true);

		// Calculate rate for fading in loading screen and progress bar.
		if (loadingScreenFadeInTime > 0.0f)
		{
			fadeRate = 1.0f / loadingScreenFadeInTime;
		}
		else
		{
			alpha = 1.0f;
		}

		var overlay = OpenVR.Overlay;

		// Optionally create our loading screen overlay.
		if (loadingScreen != null && overlay != null)
		{
			loadingScreenOverlayHandle = GetOverlayHandle("loadingScreen", loadingScreenTransform != null ? loadingScreenTransform : transform, loadingScreenWidthInMeters);
			if (loadingScreenOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
			{
				var texture = new Texture_t();
				texture.handle = loadingScreen.GetNativeTexturePtr();
				texture.eType = SteamVR.instance.textureType;
				texture.eColorSpace = EColorSpace.Auto;
				overlay.SetOverlayTexture(loadingScreenOverlayHandle, ref texture);
			}
		}

		bool fadedForeground = false;

		// Fade out to compositor
		SteamVR_Events.LoadingFadeOut.Send(fadeOutTime);

		// Optionally set a skybox to use as a backdrop in the compositor.
		var compositor = OpenVR.Compositor;
		if (compositor != null)
		{
			if (front != null)
			{
				SteamVR_Skybox.SetOverride(front, back, left, right, top, bottom);

				// Explicitly fade to the compositor since loading will cause us to stop rendering.
				compositor.FadeGrid(fadeOutTime, true);
				yield return new WaitForSeconds(fadeOutTime);
			}
			else if (backgroundColor != Color.clear)
			{
				// Otherwise, use the specified background color.
				if (showGrid)
				{
					// Set compositor background color immediately, and start fading to it.
					compositor.FadeToColor(0.0f, backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a, true);
					compositor.FadeGrid(fadeOutTime, true);
					yield return new WaitForSeconds(fadeOutTime);
				}
				else
				{
					// Fade the foreground color in (which will blend on top of the scene), and then cut to the compositor.
					compositor.FadeToColor(fadeOutTime, backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a, false);
					yield return new WaitForSeconds(fadeOutTime + 0.1f);
					compositor.FadeGrid(0.0f, true);
					fadedForeground = true;
				}
			}
		}

		// Now that we're fully faded out, we can stop submitting frames to the compositor.
		SteamVR_Render.pauseRendering = true;

		// Continue waiting for the overlays to fully fade in before continuing.
		while (alpha < 1.0f)
			yield return null;

		// Keep us from getting destroyed when loading the new level, otherwise this coroutine will get stopped prematurely.
		transform.parent = null;
		DontDestroyOnLoad(gameObject);

		if (!string.IsNullOrEmpty(internalProcessPath))
		{
			Debug.Log("Launching external application...");
			var applications = OpenVR.Applications;
			if (applications == null)
			{
				Debug.Log("Failed to get OpenVR.Applications interface!");
			}
			else
			{
				var workingDirectory = Directory.GetCurrentDirectory();
				var fullPath = Path.Combine(workingDirectory, internalProcessPath);
				Debug.Log("LaunchingInternalProcess");
				Debug.Log("ExternalAppPath = " + internalProcessPath);
				Debug.Log("FullPath = " + fullPath);
				Debug.Log("ExternalAppArgs = " + internalProcessArgs);
				Debug.Log("WorkingDirectory = " + workingDirectory);
				var error = applications.LaunchInternalProcess(fullPath, internalProcessArgs, workingDirectory);
				Debug.Log("LaunchInternalProcessError: " + error);
#if UNITY_EDITOR
				UnityEditor.EditorApplication.isPlaying = false;
#elif !UNITY_METRO
				System.Diagnostics.Process.GetCurrentProcess().Kill();
#endif
			}
		}
		else
		{
			var mode = loadAdditive ? UnityEngine.SceneManagement.LoadSceneMode.Additive : UnityEngine.SceneManagement.LoadSceneMode.Single;
			if (loadAsync)
			{
				Application.backgroundLoadingPriority = ThreadPriority.Low;
				async = UnityEngine.SceneManagement.SceneManager.LoadSceneAsync(levelName, mode);

				// Performing this in a while loop instead seems to help smooth things out.
				//yield return async;
				while (!async.isDone)
				{
					yield return null;
				}
			}
			else
			{
				UnityEngine.SceneManagement.SceneManager.LoadScene(levelName, mode);
			}
		}

		yield return null;

		System.GC.Collect();

		yield return null;

		Shader.WarmupAllShaders();

		// Optionally wait a short period of time after loading everything back in, but before we start rendering again
		// in order to give everything a change to settle down to avoid any hitching at the start of the new level.
		yield return new WaitForSeconds(postLoadSettleTime);

		SteamVR_Render.pauseRendering = false;

		// Fade out loading screen.
		if (loadingScreenFadeOutTime > 0.0f)
		{
			fadeRate = -1.0f / loadingScreenFadeOutTime;
		}
		else
		{
			alpha = 0.0f;
		}

		// Fade out to compositor
		SteamVR_Events.LoadingFadeIn.Send(fadeInTime);

		if (compositor != null)
		{
			// Fade out foreground color if necessary.
			if (fadedForeground)
			{
				compositor.FadeGrid(0.0f, false);
				compositor.FadeToColor(fadeInTime, 0.0f, 0.0f, 0.0f, 0.0f, false);
				yield return new WaitForSeconds(fadeInTime);
			}
			else
			{
				// Fade scene back in, and reset skybox once no longer visible.
				compositor.FadeGrid(fadeInTime, false);
				yield return new WaitForSeconds(fadeInTime);

				if (front != null)
				{
					SteamVR_Skybox.ClearOverride();
				}
			}
		}

		// Finally, stick around long enough for our overlays to fully fade out.
		while (alpha > 0.0f)
			yield return null;

		if (overlay != null)
		{
			if (progressBarOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
				overlay.HideOverlay(progressBarOverlayHandle);
			if (loadingScreenOverlayHandle != OpenVR.k_ulOverlayHandleInvalid)
				overlay.HideOverlay(loadingScreenOverlayHandle);
		}

		Destroy(gameObject);

		_active = null;

		SteamVR_Events.Loading.Send(false);
	}

	// Helper to create (or reuse if possible) each of our different overlay types.
	ulong GetOverlayHandle(string overlayName, Transform transform, float widthInMeters = 1.0f)
	{
		ulong handle = OpenVR.k_ulOverlayHandleInvalid;

		var overlay = OpenVR.Overlay;
		if (overlay == null)
			return handle;

		var key = SteamVR_Overlay.key + "." + overlayName;

		var error = overlay.FindOverlay(key, ref handle);
		if (error != EVROverlayError.None)
			error = overlay.CreateOverlay(key, overlayName, ref handle);
		if (error == EVROverlayError.None)
		{
			overlay.ShowOverlay(handle);
			overlay.SetOverlayAlpha(handle, alpha);
			overlay.SetOverlayWidthInMeters(handle, widthInMeters);

			// D3D textures are upside-down in Unity to match OpenGL.
			if (SteamVR.instance.textureType == ETextureType.DirectX)
			{
				var textureBounds = new VRTextureBounds_t();
				textureBounds.uMin = 0;
				textureBounds.vMin = 1;
				textureBounds.uMax = 1;
				textureBounds.vMax = 0;
				overlay.SetOverlayTextureBounds(handle, ref textureBounds);
			}

			// Convert from world space to tracking space using the top-most camera.
			var vrcam = (loadingScreenDistance == 0.0f) ? SteamVR_Render.Top() : null;
			if (vrcam != null && vrcam.origin != null)
			{
				var offset = new SteamVR_Utils.RigidTransform(vrcam.origin, transform);
				offset.pos.x /= vrcam.origin.localScale.x;
				offset.pos.y /= vrcam.origin.localScale.y;
				offset.pos.z /= vrcam.origin.localScale.z;

				var t = offset.ToHmdMatrix34();
				overlay.SetOverlayTransformAbsolute(handle, SteamVR_Render.instance.trackingSpace, ref t);
			}
			else
			{
				var t = new SteamVR_Utils.RigidTransform(transform).ToHmdMatrix34();
				overlay.SetOverlayTransformAbsolute(handle, SteamVR_Render.instance.trackingSpace, ref t);
			}
		}

		return handle;
	}
}

