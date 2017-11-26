//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Prompt developers to use settings most compatible with SteamVR.
//
//=============================================================================

using UnityEngine;
using UnityEditor;
using System.IO;

[InitializeOnLoad]
public class SteamVR_Settings : EditorWindow
{
	const bool forceShow = false; // Set to true to get the dialog to show back up in the case you clicked Ignore All.

	const string ignore = "ignore.";
	const string useRecommended = "Use recommended ({0})";
	const string currentValue = " (current = {0})";

	const string buildTarget = "Build Target";
	const string showUnitySplashScreen = "Show Unity Splashscreen";
	const string defaultIsFullScreen = "Default is Fullscreen";
	const string defaultScreenSize = "Default Screen Size";
	const string runInBackground = "Run In Background";
	const string displayResolutionDialog = "Display Resolution Dialog";
	const string resizableWindow = "Resizable Window";
	const string fullscreenMode = "D3D11 Fullscreen Mode";
	const string visibleInBackground = "Visible In Background";
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
	const string renderingPath = "Rendering Path";
#endif
	const string colorSpace = "Color Space";
	const string gpuSkinning = "GPU Skinning";
#if false // skyboxes are currently broken
	const string singlePassStereoRendering = "Single-Pass Stereo Rendering";
#endif

	const BuildTarget recommended_BuildTarget = BuildTarget.StandaloneWindows64;
	const bool recommended_ShowUnitySplashScreen = false;
	const bool recommended_DefaultIsFullScreen = false;
	const int recommended_DefaultScreenWidth = 1024;
	const int recommended_DefaultScreenHeight = 768;
	const bool recommended_RunInBackground = true;
	const ResolutionDialogSetting recommended_DisplayResolutionDialog = ResolutionDialogSetting.HiddenByDefault;
	const bool recommended_ResizableWindow = true;
	const D3D11FullscreenMode recommended_FullscreenMode = D3D11FullscreenMode.FullscreenWindow;
	const bool recommended_VisibleInBackground = true;
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
	const RenderingPath recommended_RenderPath = RenderingPath.Forward;
#endif
	const ColorSpace recommended_ColorSpace = ColorSpace.Linear;
	const bool recommended_GpuSkinning = true;
#if false
	const bool recommended_SinglePassStereoRendering = true;
#endif

	static SteamVR_Settings window;

	static SteamVR_Settings()
	{
		EditorApplication.update += Update;
	}

	static void Update()
	{
		bool show =
			(!EditorPrefs.HasKey(ignore + buildTarget) &&
				EditorUserBuildSettings.activeBuildTarget != recommended_BuildTarget) ||
			(!EditorPrefs.HasKey(ignore + showUnitySplashScreen) &&
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
				PlayerSettings.showUnitySplashScreen != recommended_ShowUnitySplashScreen) ||
#else
				PlayerSettings.SplashScreen.show != recommended_ShowUnitySplashScreen) ||
#endif
			(!EditorPrefs.HasKey(ignore + defaultIsFullScreen) &&
				PlayerSettings.defaultIsFullScreen != recommended_DefaultIsFullScreen) ||
			(!EditorPrefs.HasKey(ignore + defaultScreenSize) &&
				(PlayerSettings.defaultScreenWidth != recommended_DefaultScreenWidth ||
				PlayerSettings.defaultScreenHeight != recommended_DefaultScreenHeight)) ||
			(!EditorPrefs.HasKey(ignore + runInBackground) &&
				PlayerSettings.runInBackground != recommended_RunInBackground) ||
			(!EditorPrefs.HasKey(ignore + displayResolutionDialog) &&
				PlayerSettings.displayResolutionDialog != recommended_DisplayResolutionDialog) ||
			(!EditorPrefs.HasKey(ignore + resizableWindow) &&
				PlayerSettings.resizableWindow != recommended_ResizableWindow) ||
			(!EditorPrefs.HasKey(ignore + fullscreenMode) &&
				PlayerSettings.d3d11FullscreenMode != recommended_FullscreenMode) ||
			(!EditorPrefs.HasKey(ignore + visibleInBackground) &&
				PlayerSettings.visibleInBackground != recommended_VisibleInBackground) ||
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
			(!EditorPrefs.HasKey(ignore + renderingPath) &&
				PlayerSettings.renderingPath != recommended_RenderPath) ||
#endif
			(!EditorPrefs.HasKey(ignore + colorSpace) &&
				PlayerSettings.colorSpace != recommended_ColorSpace) ||
			(!EditorPrefs.HasKey(ignore + gpuSkinning) &&
				PlayerSettings.gpuSkinning != recommended_GpuSkinning) ||
#if false
			(!EditorPrefs.HasKey(ignore + singlePassStereoRendering) &&
				PlayerSettings.singlePassStereoRendering != recommended_SinglePassStereoRendering) ||
#endif
			forceShow;

		if (show)
		{
			window = GetWindow<SteamVR_Settings>(true);
			window.minSize = new Vector2(320, 440);
			//window.title = "SteamVR";
		}

		if (SteamVR_Preferences.AutoEnableVR)
		{
			// Switch to native OpenVR support.
			var updated = false;

			if (!PlayerSettings.virtualRealitySupported)
			{
				PlayerSettings.virtualRealitySupported = true;
				updated = true;
			}

#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
			var devices = UnityEditorInternal.VR.VREditor.GetVREnabledDevices(BuildTargetGroup.Standalone);
#else
			var devices = UnityEditorInternal.VR.VREditor.GetVREnabledDevicesOnTargetGroup(BuildTargetGroup.Standalone);
#endif
			var hasOpenVR = false;
			foreach (var device in devices)
				if (device.ToLower() == "openvr")
					hasOpenVR = true;


			if (!hasOpenVR)
			{
				string[] newDevices;
				if (updated)
				{
					newDevices = new string[] { "OpenVR" };
				}
				else
				{
					newDevices = new string[devices.Length + 1];
					for (int i = 0; i < devices.Length; i++)
						newDevices[i] = devices[i];
					newDevices[devices.Length] = "OpenVR";
					updated = true;
				}
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
				UnityEditorInternal.VR.VREditor.SetVREnabledDevices(BuildTargetGroup.Standalone, newDevices);
#else
				UnityEditorInternal.VR.VREditor.SetVREnabledDevicesOnTargetGroup(BuildTargetGroup.Standalone, newDevices);
#endif
			}

			if (updated)
				Debug.Log("Switching to native OpenVR support.");
		}

		var dlls = new string[]
		{
			"Plugins/x86/openvr_api.dll",
			"Plugins/x86_64/openvr_api.dll"
		};

		foreach (var path in dlls)
		{
			if (!File.Exists(Application.dataPath + "/" + path))
				continue;

			if (AssetDatabase.DeleteAsset("Assets/" + path))
				Debug.Log("Deleting " + path);
			else
			{
				Debug.Log(path + " in use; cannot delete.  Please restart Unity to complete upgrade.");
			}
		}

		EditorApplication.update -= Update;
	}

	Vector2 scrollPosition;
	
	string GetResourcePath()
	{
		var ms = MonoScript.FromScriptableObject(this);
		var path = AssetDatabase.GetAssetPath(ms);
		path = Path.GetDirectoryName(path);
		return path.Substring(0, path.Length - "Editor".Length) + "Textures/";
	}

	public void OnGUI()
	{
		var resourcePath = GetResourcePath();
		var logo = AssetDatabase.LoadAssetAtPath<Texture2D>(resourcePath + "logo.png");
		var rect = GUILayoutUtility.GetRect(position.width, 150, GUI.skin.box);
		if (logo)
			GUI.DrawTexture(rect, logo, ScaleMode.ScaleToFit);

		EditorGUILayout.HelpBox("Recommended project settings for SteamVR:", MessageType.Warning);

		scrollPosition = GUILayout.BeginScrollView(scrollPosition);

		int numItems = 0;

		if (!EditorPrefs.HasKey(ignore + buildTarget) &&
			EditorUserBuildSettings.activeBuildTarget != recommended_BuildTarget)
		{
			++numItems;

			GUILayout.Label(buildTarget + string.Format(currentValue, EditorUserBuildSettings.activeBuildTarget));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_BuildTarget)))
			{
#if (UNITY_5_5 || UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
				EditorUserBuildSettings.SwitchActiveBuildTarget(recommended_BuildTarget);
#else
				EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, recommended_BuildTarget);
#endif
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + buildTarget, true);
			}

			GUILayout.EndHorizontal();
		}

#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
		if (!EditorPrefs.HasKey(ignore + showUnitySplashScreen) &&
			PlayerSettings.showUnitySplashScreen != recommended_ShowUnitySplashScreen)
		{
			++numItems;

			GUILayout.Label(showUnitySplashScreen + string.Format(currentValue, PlayerSettings.showUnitySplashScreen));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_ShowUnitySplashScreen)))
			{
				PlayerSettings.showUnitySplashScreen = recommended_ShowUnitySplashScreen;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + showUnitySplashScreen, true);
			}

			GUILayout.EndHorizontal();
		}
#else
		if (!EditorPrefs.HasKey(ignore + showUnitySplashScreen) &&
			PlayerSettings.SplashScreen.show != recommended_ShowUnitySplashScreen)
		{
			++numItems;

			GUILayout.Label(showUnitySplashScreen + string.Format(currentValue, PlayerSettings.SplashScreen.show));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_ShowUnitySplashScreen)))
			{
				PlayerSettings.SplashScreen.show = recommended_ShowUnitySplashScreen;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + showUnitySplashScreen, true);
			}

			GUILayout.EndHorizontal();
		}
#endif
		if (!EditorPrefs.HasKey(ignore + defaultIsFullScreen) &&
			PlayerSettings.defaultIsFullScreen != recommended_DefaultIsFullScreen)
		{
			++numItems;

			GUILayout.Label(defaultIsFullScreen + string.Format(currentValue, PlayerSettings.defaultIsFullScreen));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_DefaultIsFullScreen)))
			{
				PlayerSettings.defaultIsFullScreen = recommended_DefaultIsFullScreen;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + defaultIsFullScreen, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + defaultScreenSize) &&
			(PlayerSettings.defaultScreenWidth != recommended_DefaultScreenWidth ||
			PlayerSettings.defaultScreenHeight != recommended_DefaultScreenHeight))
		{
			++numItems;

			GUILayout.Label(defaultScreenSize + string.Format(" ({0}x{1})", PlayerSettings.defaultScreenWidth, PlayerSettings.defaultScreenHeight));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format("Use recommended ({0}x{1})", recommended_DefaultScreenWidth, recommended_DefaultScreenHeight)))
			{
				PlayerSettings.defaultScreenWidth = recommended_DefaultScreenWidth;
				PlayerSettings.defaultScreenHeight = recommended_DefaultScreenHeight;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + defaultScreenSize, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + runInBackground) &&
			PlayerSettings.runInBackground != recommended_RunInBackground)
		{
			++numItems;

			GUILayout.Label(runInBackground + string.Format(currentValue, PlayerSettings.runInBackground));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_RunInBackground)))
			{
				PlayerSettings.runInBackground = recommended_RunInBackground;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + runInBackground, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + displayResolutionDialog) &&
			PlayerSettings.displayResolutionDialog != recommended_DisplayResolutionDialog)
		{
			++numItems;

			GUILayout.Label(displayResolutionDialog + string.Format(currentValue, PlayerSettings.displayResolutionDialog));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_DisplayResolutionDialog)))
			{
				PlayerSettings.displayResolutionDialog = recommended_DisplayResolutionDialog;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + displayResolutionDialog, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + resizableWindow) &&
			PlayerSettings.resizableWindow != recommended_ResizableWindow)
		{
			++numItems;

			GUILayout.Label(resizableWindow + string.Format(currentValue, PlayerSettings.resizableWindow));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_ResizableWindow)))
			{
				PlayerSettings.resizableWindow = recommended_ResizableWindow;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + resizableWindow, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + fullscreenMode) &&
			PlayerSettings.d3d11FullscreenMode != recommended_FullscreenMode)
		{
			++numItems;

			GUILayout.Label(fullscreenMode + string.Format(currentValue, PlayerSettings.d3d11FullscreenMode));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_FullscreenMode)))
			{
				PlayerSettings.d3d11FullscreenMode = recommended_FullscreenMode;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + fullscreenMode, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + visibleInBackground) &&
			PlayerSettings.visibleInBackground != recommended_VisibleInBackground)
		{
			++numItems;

			GUILayout.Label(visibleInBackground + string.Format(currentValue, PlayerSettings.visibleInBackground));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_VisibleInBackground)))
			{
				PlayerSettings.visibleInBackground = recommended_VisibleInBackground;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + visibleInBackground, true);
			}

			GUILayout.EndHorizontal();
		}
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
		if (!EditorPrefs.HasKey(ignore + renderingPath) &&
			PlayerSettings.renderingPath != recommended_RenderPath)
		{
			++numItems;

			GUILayout.Label(renderingPath + string.Format(currentValue, PlayerSettings.renderingPath));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_RenderPath) + " - required for MSAA"))
			{
				PlayerSettings.renderingPath = recommended_RenderPath;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + renderingPath, true);
			}

			GUILayout.EndHorizontal();
		}
#endif
		if (!EditorPrefs.HasKey(ignore + colorSpace) &&
			PlayerSettings.colorSpace != recommended_ColorSpace)
		{
			++numItems;

			GUILayout.Label(colorSpace + string.Format(currentValue, PlayerSettings.colorSpace));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_ColorSpace) + " - requires reloading scene"))
			{
				PlayerSettings.colorSpace = recommended_ColorSpace;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + colorSpace, true);
			}

			GUILayout.EndHorizontal();
		}

		if (!EditorPrefs.HasKey(ignore + gpuSkinning) &&
			PlayerSettings.gpuSkinning != recommended_GpuSkinning)
		{
			++numItems;

			GUILayout.Label(gpuSkinning + string.Format(currentValue, PlayerSettings.gpuSkinning));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_GpuSkinning)))
			{
				PlayerSettings.gpuSkinning = recommended_GpuSkinning;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + gpuSkinning, true);
			}

			GUILayout.EndHorizontal();
		}

#if false
		if (!EditorPrefs.HasKey(ignore + singlePassStereoRendering) &&
			PlayerSettings.singlePassStereoRendering != recommended_SinglePassStereoRendering)
		{
			++numItems;

			GUILayout.Label(singlePassStereoRendering + string.Format(currentValue, PlayerSettings.singlePassStereoRendering));

			GUILayout.BeginHorizontal();

			if (GUILayout.Button(string.Format(useRecommended, recommended_SinglePassStereoRendering)))
			{
				PlayerSettings.singlePassStereoRendering = recommended_SinglePassStereoRendering;
			}

			GUILayout.FlexibleSpace();

			if (GUILayout.Button("Ignore"))
			{
				EditorPrefs.SetBool(ignore + singlePassStereoRendering, true);
			}

			GUILayout.EndHorizontal();
		}
#endif

		GUILayout.BeginHorizontal();

		GUILayout.FlexibleSpace();

		if (GUILayout.Button("Clear All Ignores"))
		{
			EditorPrefs.DeleteKey(ignore + buildTarget);
			EditorPrefs.DeleteKey(ignore + showUnitySplashScreen);
			EditorPrefs.DeleteKey(ignore + defaultIsFullScreen);
			EditorPrefs.DeleteKey(ignore + defaultScreenSize);
			EditorPrefs.DeleteKey(ignore + runInBackground);
			EditorPrefs.DeleteKey(ignore + displayResolutionDialog);
			EditorPrefs.DeleteKey(ignore + resizableWindow);
			EditorPrefs.DeleteKey(ignore + fullscreenMode);
			EditorPrefs.DeleteKey(ignore + visibleInBackground);
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
			EditorPrefs.DeleteKey(ignore + renderingPath);
#endif
			EditorPrefs.DeleteKey(ignore + colorSpace);
			EditorPrefs.DeleteKey(ignore + gpuSkinning);
#if false
			EditorPrefs.DeleteKey(ignore + singlePassStereoRendering);
#endif
		}

		GUILayout.EndHorizontal();

		GUILayout.EndScrollView();

		GUILayout.FlexibleSpace();

		GUILayout.BeginHorizontal();

		if (numItems > 0)
		{
			if (GUILayout.Button("Accept All"))
			{
				// Only set those that have not been explicitly ignored.
				if (!EditorPrefs.HasKey(ignore + buildTarget))
#if (UNITY_5_5 || UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					EditorUserBuildSettings.SwitchActiveBuildTarget(recommended_BuildTarget);
#else
					EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, recommended_BuildTarget);
#endif
				if (!EditorPrefs.HasKey(ignore + showUnitySplashScreen))
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					PlayerSettings.showUnitySplashScreen = recommended_ShowUnitySplashScreen;
#else
					PlayerSettings.SplashScreen.show = recommended_ShowUnitySplashScreen;
#endif
				if (!EditorPrefs.HasKey(ignore + defaultIsFullScreen))
					PlayerSettings.defaultIsFullScreen = recommended_DefaultIsFullScreen;
				if (!EditorPrefs.HasKey(ignore + defaultScreenSize))
				{
					PlayerSettings.defaultScreenWidth = recommended_DefaultScreenWidth;
					PlayerSettings.defaultScreenHeight = recommended_DefaultScreenHeight;
				}
				if (!EditorPrefs.HasKey(ignore + runInBackground))
					PlayerSettings.runInBackground = recommended_RunInBackground;
				if (!EditorPrefs.HasKey(ignore + displayResolutionDialog))
					PlayerSettings.displayResolutionDialog = recommended_DisplayResolutionDialog;
				if (!EditorPrefs.HasKey(ignore + resizableWindow))
					PlayerSettings.resizableWindow = recommended_ResizableWindow;
				if (!EditorPrefs.HasKey(ignore + fullscreenMode))
					PlayerSettings.d3d11FullscreenMode = recommended_FullscreenMode;
				if (!EditorPrefs.HasKey(ignore + visibleInBackground))
					PlayerSettings.visibleInBackground = recommended_VisibleInBackground;
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
				if (!EditorPrefs.HasKey(ignore + renderingPath))
					PlayerSettings.renderingPath = recommended_RenderPath;
#endif
				if (!EditorPrefs.HasKey(ignore + colorSpace))
					PlayerSettings.colorSpace = recommended_ColorSpace;
				if (!EditorPrefs.HasKey(ignore + gpuSkinning))
					PlayerSettings.gpuSkinning = recommended_GpuSkinning;
#if false
				if (!EditorPrefs.HasKey(ignore + singlePassStereoRendering))
					PlayerSettings.singlePassStereoRendering = recommended_SinglePassStereoRendering;
#endif

				EditorUtility.DisplayDialog("Accept All", "You made the right choice!", "Ok");

				Close();
			}

			if (GUILayout.Button("Ignore All"))
			{
				if (EditorUtility.DisplayDialog("Ignore All", "Are you sure?", "Yes, Ignore All", "Cancel"))
				{
					// Only ignore those that do not currently match our recommended settings.
					if (EditorUserBuildSettings.activeBuildTarget != recommended_BuildTarget)
						EditorPrefs.SetBool(ignore + buildTarget, true);
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					if (PlayerSettings.showUnitySplashScreen != recommended_ShowUnitySplashScreen)
#else
					if (PlayerSettings.SplashScreen.show != recommended_ShowUnitySplashScreen)
#endif
						EditorPrefs.SetBool(ignore + showUnitySplashScreen, true);
					if (PlayerSettings.defaultIsFullScreen != recommended_DefaultIsFullScreen)
						EditorPrefs.SetBool(ignore + defaultIsFullScreen, true);
					if (PlayerSettings.defaultScreenWidth != recommended_DefaultScreenWidth ||
						PlayerSettings.defaultScreenHeight != recommended_DefaultScreenHeight)
						EditorPrefs.SetBool(ignore + defaultScreenSize, true);
					if (PlayerSettings.runInBackground != recommended_RunInBackground)
						EditorPrefs.SetBool(ignore + runInBackground, true);
					if (PlayerSettings.displayResolutionDialog != recommended_DisplayResolutionDialog)
						EditorPrefs.SetBool(ignore + displayResolutionDialog, true);
					if (PlayerSettings.resizableWindow != recommended_ResizableWindow)
						EditorPrefs.SetBool(ignore + resizableWindow, true);
					if (PlayerSettings.d3d11FullscreenMode != recommended_FullscreenMode)
						EditorPrefs.SetBool(ignore + fullscreenMode, true);
					if (PlayerSettings.visibleInBackground != recommended_VisibleInBackground)
						EditorPrefs.SetBool(ignore + visibleInBackground, true);
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					if (PlayerSettings.renderingPath != recommended_RenderPath)
						EditorPrefs.SetBool(ignore + renderingPath, true);
#endif
					if (PlayerSettings.colorSpace != recommended_ColorSpace)
						EditorPrefs.SetBool(ignore + colorSpace, true);
					if (PlayerSettings.gpuSkinning != recommended_GpuSkinning)
						EditorPrefs.SetBool(ignore + gpuSkinning, true);
#if false
					if (PlayerSettings.singlePassStereoRendering != recommended_SinglePassStereoRendering)
						EditorPrefs.SetBool(ignore + singlePassStereoRendering, true);
#endif

					Close();
				}
			}
		}
		else if (GUILayout.Button("Close"))
		{
			Close();
		}

		GUILayout.EndHorizontal();
	}
}

