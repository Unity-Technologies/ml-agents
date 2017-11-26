//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Access to SteamVR system (hmd) and compositor (distort) interfaces.
//
//=============================================================================

using UnityEngine;
using Valve.VR;

public class SteamVR : System.IDisposable
{
	// Use this to check if SteamVR is currently active without attempting
	// to activate it in the process.
	public static bool active { get { return _instance != null; } }

	// Set this to false to keep from auto-initializing when calling SteamVR.instance.
	private static bool _enabled = true;
	public static bool enabled
	{
		get
		{
			if (!UnityEngine.VR.VRSettings.enabled)
				enabled = false;
			return _enabled;
		}
		set
		{
			_enabled = value;
			if (!_enabled)
				SafeDispose();
		}
	}

	private static SteamVR _instance;
	public static SteamVR instance
	{
		get
		{
#if UNITY_EDITOR
			if (!Application.isPlaying)
				return null;
#endif
			if (!enabled)
				return null;

			if (_instance == null)
			{
				_instance = CreateInstance();

				// If init failed, then auto-disable so scripts don't continue trying to re-initialize things.
				if (_instance == null)
					_enabled = false;
			}

			return _instance;
		}
	}

	public static bool usingNativeSupport
	{
		get { return UnityEngine.VR.VRDevice.GetNativePtr() != System.IntPtr.Zero; }
	}

	static SteamVR CreateInstance()
	{
		try
		{
			var error = EVRInitError.None;
			if (!SteamVR.usingNativeSupport)
			{
				Debug.Log("OpenVR initialization failed.  Ensure 'Virtual Reality Supported' is checked in Player Settings, and OpenVR is added to the list of Virtual Reality SDKs.");
				return null;
			}

			// Verify common interfaces are valid.

			OpenVR.GetGenericInterface(OpenVR.IVRCompositor_Version, ref error);
			if (error != EVRInitError.None)
			{
				ReportError(error);
				return null;
			}

			OpenVR.GetGenericInterface(OpenVR.IVROverlay_Version, ref error);
			if (error != EVRInitError.None)
			{
				ReportError(error);
				return null;
			}
		}
		catch (System.Exception e)
		{
			Debug.LogError(e);
			return null;
		}

		return new SteamVR();
	}

	static void ReportError(EVRInitError error)
	{
		switch (error)
		{
			case EVRInitError.None:
				break;
			case EVRInitError.VendorSpecific_UnableToConnectToOculusRuntime:
				Debug.Log("SteamVR Initialization Failed!  Make sure device is on, Oculus runtime is installed, and OVRService_*.exe is running.");
				break;
			case EVRInitError.Init_VRClientDLLNotFound:
				Debug.Log("SteamVR drivers not found!  They can be installed via Steam under Library > Tools.  Visit http://steampowered.com to install Steam.");
				break;
			case EVRInitError.Driver_RuntimeOutOfDate:
				Debug.Log("SteamVR Initialization Failed!  Make sure device's runtime is up to date.");
				break;
			default:
				Debug.Log(OpenVR.GetStringForHmdError(error));
				break;
		}
	}

	// native interfaces
	public CVRSystem hmd { get; private set; }
	public CVRCompositor compositor { get; private set; }
	public CVROverlay overlay { get; private set; }

	// tracking status
	static public bool initializing { get; private set; }
	static public bool calibrating { get; private set; }
	static public bool outOfRange { get; private set; }

	static public bool[] connected = new bool[OpenVR.k_unMaxTrackedDeviceCount];

	// render values
	public float sceneWidth { get; private set; }
	public float sceneHeight { get; private set; }
	public float aspect { get; private set; }
	public float fieldOfView { get; private set; }
	public Vector2 tanHalfFov { get; private set; }
	public VRTextureBounds_t[] textureBounds { get; private set; }
	public SteamVR_Utils.RigidTransform[] eyes { get; private set; }
	public ETextureType textureType;

	// hmd properties
	public string hmd_TrackingSystemName { get { return GetStringProperty(ETrackedDeviceProperty.Prop_TrackingSystemName_String); } }
	public string hmd_ModelNumber { get { return GetStringProperty(ETrackedDeviceProperty.Prop_ModelNumber_String); } }
	public string hmd_SerialNumber { get { return GetStringProperty(ETrackedDeviceProperty.Prop_SerialNumber_String); } }

	public float hmd_SecondsFromVsyncToPhotons { get { return GetFloatProperty(ETrackedDeviceProperty.Prop_SecondsFromVsyncToPhotons_Float); } }
	public float hmd_DisplayFrequency { get { return GetFloatProperty(ETrackedDeviceProperty.Prop_DisplayFrequency_Float); } }

	public string GetTrackedDeviceString(uint deviceId)
	{
		var error = ETrackedPropertyError.TrackedProp_Success;
		var capacity = hmd.GetStringTrackedDeviceProperty(deviceId, ETrackedDeviceProperty.Prop_AttachedDeviceId_String, null, 0, ref error);
		if (capacity > 1)
		{
			var result = new System.Text.StringBuilder((int)capacity);
			hmd.GetStringTrackedDeviceProperty(deviceId, ETrackedDeviceProperty.Prop_AttachedDeviceId_String, result, capacity, ref error);
			return result.ToString();
		}
		return null;
	}

	public string GetStringProperty(ETrackedDeviceProperty prop, uint deviceId = OpenVR.k_unTrackedDeviceIndex_Hmd)
	{
		var error = ETrackedPropertyError.TrackedProp_Success;
		var capactiy = hmd.GetStringTrackedDeviceProperty(deviceId, prop, null, 0, ref error);
		if (capactiy > 1)
		{
			var result = new System.Text.StringBuilder((int)capactiy);
			hmd.GetStringTrackedDeviceProperty(deviceId, prop, result, capactiy, ref error);
			return result.ToString();
		}
		return (error != ETrackedPropertyError.TrackedProp_Success) ? error.ToString() : "<unknown>";
	}

	public float GetFloatProperty(ETrackedDeviceProperty prop, uint deviceId = OpenVR.k_unTrackedDeviceIndex_Hmd)
	{
		var error = ETrackedPropertyError.TrackedProp_Success;
		return hmd.GetFloatTrackedDeviceProperty(deviceId, prop, ref error);
	}

	#region Event callbacks

	private void OnInitializing(bool initializing)
	{
		SteamVR.initializing = initializing;
	}

	private void OnCalibrating(bool calibrating)
	{
		SteamVR.calibrating = calibrating;
	}

	private void OnOutOfRange(bool outOfRange)
	{
		SteamVR.outOfRange = outOfRange;
	}

	private void OnDeviceConnected(int i, bool connected)
	{
		SteamVR.connected[i] = connected;
	}

	private void OnNewPoses(TrackedDevicePose_t[] poses)
	{
		// Update eye offsets to account for IPD changes.
		eyes[0] = new SteamVR_Utils.RigidTransform(hmd.GetEyeToHeadTransform(EVREye.Eye_Left));
		eyes[1] = new SteamVR_Utils.RigidTransform(hmd.GetEyeToHeadTransform(EVREye.Eye_Right));

		for (int i = 0; i < poses.Length; i++)
		{
			var connected = poses[i].bDeviceIsConnected;
			if (connected != SteamVR.connected[i])
			{
				SteamVR_Events.DeviceConnected.Send(i, connected);
			}
		}

		if (poses.Length > OpenVR.k_unTrackedDeviceIndex_Hmd)
		{
			var result = poses[OpenVR.k_unTrackedDeviceIndex_Hmd].eTrackingResult;

			var initializing = result == ETrackingResult.Uninitialized;
			if (initializing != SteamVR.initializing)
			{
				SteamVR_Events.Initializing.Send(initializing);
			}

			var calibrating =
				result == ETrackingResult.Calibrating_InProgress ||
				result == ETrackingResult.Calibrating_OutOfRange;
			if (calibrating != SteamVR.calibrating)
			{
				SteamVR_Events.Calibrating.Send(calibrating);
			}

			var outOfRange =
				result == ETrackingResult.Running_OutOfRange ||
				result == ETrackingResult.Calibrating_OutOfRange;
			if (outOfRange != SteamVR.outOfRange)
			{
				SteamVR_Events.OutOfRange.Send(outOfRange);
			}
		}
	}

	#endregion

	private SteamVR()
	{
		hmd = OpenVR.System;
		Debug.Log("Connected to " + hmd_TrackingSystemName + ":" + hmd_SerialNumber);

		compositor = OpenVR.Compositor;
		overlay = OpenVR.Overlay;

		// Setup render values
		uint w = 0, h = 0;
		hmd.GetRecommendedRenderTargetSize(ref w, ref h);
		sceneWidth = (float)w;
		sceneHeight = (float)h;

		float l_left = 0.0f, l_right = 0.0f, l_top = 0.0f, l_bottom = 0.0f;
		hmd.GetProjectionRaw(EVREye.Eye_Left, ref l_left, ref l_right, ref l_top, ref l_bottom);

		float r_left = 0.0f, r_right = 0.0f, r_top = 0.0f, r_bottom = 0.0f;
		hmd.GetProjectionRaw(EVREye.Eye_Right, ref r_left, ref r_right, ref r_top, ref r_bottom);

		tanHalfFov = new Vector2(
			Mathf.Max(-l_left, l_right, -r_left, r_right),
			Mathf.Max(-l_top, l_bottom, -r_top, r_bottom));

		textureBounds = new VRTextureBounds_t[2];

		textureBounds[0].uMin = 0.5f + 0.5f * l_left / tanHalfFov.x;
		textureBounds[0].uMax = 0.5f + 0.5f * l_right / tanHalfFov.x;
		textureBounds[0].vMin = 0.5f - 0.5f * l_bottom / tanHalfFov.y;
		textureBounds[0].vMax = 0.5f - 0.5f * l_top / tanHalfFov.y;

		textureBounds[1].uMin = 0.5f + 0.5f * r_left / tanHalfFov.x;
		textureBounds[1].uMax = 0.5f + 0.5f * r_right / tanHalfFov.x;
		textureBounds[1].vMin = 0.5f - 0.5f * r_bottom / tanHalfFov.y;
		textureBounds[1].vMax = 0.5f - 0.5f * r_top / tanHalfFov.y;

		// Grow the recommended size to account for the overlapping fov
		sceneWidth = sceneWidth / Mathf.Max(textureBounds[0].uMax - textureBounds[0].uMin, textureBounds[1].uMax - textureBounds[1].uMin);
		sceneHeight = sceneHeight / Mathf.Max(textureBounds[0].vMax - textureBounds[0].vMin, textureBounds[1].vMax - textureBounds[1].vMin);

		aspect = tanHalfFov.x / tanHalfFov.y;
		fieldOfView = 2.0f * Mathf.Atan(tanHalfFov.y) * Mathf.Rad2Deg;

		eyes = new SteamVR_Utils.RigidTransform[] {
			new SteamVR_Utils.RigidTransform(hmd.GetEyeToHeadTransform(EVREye.Eye_Left)),
			new SteamVR_Utils.RigidTransform(hmd.GetEyeToHeadTransform(EVREye.Eye_Right)) };

		switch (SystemInfo.graphicsDeviceType)
		{
#if (UNITY_5_4)
			case UnityEngine.Rendering.GraphicsDeviceType.OpenGL2:
#endif
			case UnityEngine.Rendering.GraphicsDeviceType.OpenGLCore:
			case UnityEngine.Rendering.GraphicsDeviceType.OpenGLES2:
			case UnityEngine.Rendering.GraphicsDeviceType.OpenGLES3:
				textureType = ETextureType.OpenGL;
				break;
#if !(UNITY_5_4)
			case UnityEngine.Rendering.GraphicsDeviceType.Vulkan:
				textureType = ETextureType.Vulkan;
				break;
#endif
			default:
				textureType = ETextureType.DirectX;
				break;
		}

		SteamVR_Events.Initializing.Listen(OnInitializing);
		SteamVR_Events.Calibrating.Listen(OnCalibrating);
		SteamVR_Events.OutOfRange.Listen(OnOutOfRange);
		SteamVR_Events.DeviceConnected.Listen(OnDeviceConnected);
		SteamVR_Events.NewPoses.Listen(OnNewPoses);
	}

	~SteamVR()
	{
		Dispose(false);
	}

	public void Dispose()
	{
		Dispose(true);
		System.GC.SuppressFinalize(this);
	}

	private void Dispose(bool disposing)
	{
		SteamVR_Events.Initializing.Remove(OnInitializing);
		SteamVR_Events.Calibrating.Remove(OnCalibrating);
		SteamVR_Events.OutOfRange.Remove(OnOutOfRange);
		SteamVR_Events.DeviceConnected.Remove(OnDeviceConnected);
		SteamVR_Events.NewPoses.Remove(OnNewPoses);

		_instance = null;
	}

	// Use this interface to avoid accidentally creating the instance in the process of attempting to dispose of it.
	public static void SafeDispose()
	{
		if (_instance != null)
			_instance.Dispose();
	}
}

