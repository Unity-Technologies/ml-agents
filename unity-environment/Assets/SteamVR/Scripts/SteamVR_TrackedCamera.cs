//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Provides access to video feed and poses of tracked cameras.
//
// Usage:
//			var source = SteamVR_TrackedCamera.Distorted();
//			var source = SteamVR_TrackedCamera.Undistorted();
// or
//			var undistorted = true; // or false
//			var source = SteamVR_TrackedCamera.Source(undistorted);
//
// - Distorted feeds are the decoded images from the camera.
// - Undistorted feeds correct for the camera lens distortion (a.k.a. fisheye)
//   to make straight lines straight.
//
// VideoStreamTexture objects must be symmetrically Acquired and Released to
// ensure the video stream is activated, and shutdown properly once there are
// no more consumers.  You only need to Acquire once when starting to use a
// stream, and Release when you are done using it (as opposed to every frame).
//
//=============================================================================

using UnityEngine;
using Valve.VR;

public class SteamVR_TrackedCamera
{
	public class VideoStreamTexture
	{
		public VideoStreamTexture(uint deviceIndex, bool undistorted)
		{
			this.undistorted = undistorted;
			videostream = Stream(deviceIndex);
		}
		public bool undistorted { get; private set; }
		public uint deviceIndex { get { return videostream.deviceIndex; } }
		public bool hasCamera { get { return videostream.hasCamera; } }
		public bool hasTracking { get { Update(); return header.standingTrackedDevicePose.bPoseIsValid; } }
		public uint frameId { get { Update(); return header.nFrameSequence; } }
		public VRTextureBounds_t frameBounds { get; private set; }
		public EVRTrackedCameraFrameType frameType { get { return undistorted ? EVRTrackedCameraFrameType.Undistorted : EVRTrackedCameraFrameType.Distorted; } }

		Texture2D _texture;
		public Texture2D texture { get { Update(); return _texture; } }

		public SteamVR_Utils.RigidTransform transform { get { Update(); return new SteamVR_Utils.RigidTransform(header.standingTrackedDevicePose.mDeviceToAbsoluteTracking); } }
		public Vector3 velocity { get { Update(); var pose = header.standingTrackedDevicePose; return new Vector3(pose.vVelocity.v0, pose.vVelocity.v1, -pose.vVelocity.v2); } }
		public Vector3 angularVelocity { get { Update(); var pose = header.standingTrackedDevicePose; return new Vector3(-pose.vAngularVelocity.v0, -pose.vAngularVelocity.v1, pose.vAngularVelocity.v2); } }

		public TrackedDevicePose_t GetPose() { Update(); return header.standingTrackedDevicePose; }

		public ulong Acquire()
		{
			return videostream.Acquire();
		}
		public ulong Release()
		{
			var result = videostream.Release();

			if (videostream.handle == 0)
			{
				Object.Destroy(_texture);
				_texture = null;
			}

			return result;
		}

		int prevFrameCount = -1;
		void Update()
		{
			if (Time.frameCount == prevFrameCount)
				return;

			prevFrameCount = Time.frameCount;

			if (videostream.handle == 0)
				return;

			var vr = SteamVR.instance;
			if (vr == null)
				return;

			var trackedCamera = OpenVR.TrackedCamera;
			if (trackedCamera == null)
				return;

			var nativeTex = System.IntPtr.Zero;
			var deviceTexture = (_texture != null) ? _texture : new Texture2D(2, 2);
			var headerSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(header.GetType());

			if (vr.textureType == ETextureType.OpenGL)
			{
				if (glTextureId != 0)
					trackedCamera.ReleaseVideoStreamTextureGL(videostream.handle, glTextureId);

				if (trackedCamera.GetVideoStreamTextureGL(videostream.handle, frameType, ref glTextureId, ref header, headerSize) != EVRTrackedCameraError.None)
					return;

				nativeTex = (System.IntPtr)glTextureId;
            }
			else if (vr.textureType == ETextureType.DirectX)
			{
				if (trackedCamera.GetVideoStreamTextureD3D11(videostream.handle, frameType, deviceTexture.GetNativeTexturePtr(), ref nativeTex, ref header, headerSize) != EVRTrackedCameraError.None)
					return;
			}

			if (_texture == null)
			{
				_texture = Texture2D.CreateExternalTexture((int)header.nWidth, (int)header.nHeight, TextureFormat.RGBA32, false, false, nativeTex);

				uint width = 0, height = 0;
				var frameBounds = new VRTextureBounds_t();
				if (trackedCamera.GetVideoStreamTextureSize(deviceIndex, frameType, ref frameBounds, ref width, ref height) == EVRTrackedCameraError.None)
				{
					// Account for textures being upside-down in Unity.
					frameBounds.vMin = 1.0f - frameBounds.vMin;
					frameBounds.vMax = 1.0f - frameBounds.vMax;
					this.frameBounds = frameBounds;
				}
			}
			else
			{
				_texture.UpdateExternalTexture(nativeTex);
			}
		}

		uint glTextureId;
		VideoStream videostream;
		CameraVideoStreamFrameHeader_t header;
	}

	#region Top level accessors.

	public static VideoStreamTexture Distorted(int deviceIndex = (int)OpenVR.k_unTrackedDeviceIndex_Hmd)
	{
		if (distorted == null)
			distorted = new VideoStreamTexture[OpenVR.k_unMaxTrackedDeviceCount];
		if (distorted[deviceIndex] == null)
			distorted[deviceIndex] = new VideoStreamTexture((uint)deviceIndex, false);
		return distorted[deviceIndex];
	}

	public static VideoStreamTexture Undistorted(int deviceIndex = (int)OpenVR.k_unTrackedDeviceIndex_Hmd)
	{
		if (undistorted == null)
			undistorted = new VideoStreamTexture[OpenVR.k_unMaxTrackedDeviceCount];
		if (undistorted[deviceIndex] == null)
			undistorted[deviceIndex] = new VideoStreamTexture((uint)deviceIndex, true);
		return undistorted[deviceIndex];
	}

	public static VideoStreamTexture Source(bool undistorted, int deviceIndex = (int)OpenVR.k_unTrackedDeviceIndex_Hmd)
	{
		return undistorted ? Undistorted(deviceIndex) : Distorted(deviceIndex);
    }

	private static VideoStreamTexture[] distorted, undistorted;

	#endregion

	#region Internal class to manage lifetime of video streams (per device).

	class VideoStream
	{
		public VideoStream(uint deviceIndex)
		{
			this.deviceIndex = deviceIndex;
			var trackedCamera = OpenVR.TrackedCamera;
			if (trackedCamera != null)
				trackedCamera.HasCamera(deviceIndex, ref _hasCamera);
		}
		public uint deviceIndex { get; private set; }

		ulong _handle;
		public ulong handle { get { return _handle; } }

		bool _hasCamera;
		public bool hasCamera { get { return _hasCamera; } }

		ulong refCount;
		public ulong Acquire()
		{
			if (_handle == 0 && hasCamera)
			{
				var trackedCamera = OpenVR.TrackedCamera;
				if (trackedCamera != null)
					trackedCamera.AcquireVideoStreamingService(deviceIndex, ref _handle);
			}
			return ++refCount;
		}
		public ulong Release()
		{
			if (refCount > 0 && --refCount == 0 && _handle != 0)
			{
				var trackedCamera = OpenVR.TrackedCamera;
				if (trackedCamera != null)
					trackedCamera.ReleaseVideoStreamingService(_handle);
				_handle = 0;
			}
			return refCount;
		}
	}

	static VideoStream Stream(uint deviceIndex)
	{
		if (videostreams == null)
			videostreams = new VideoStream[OpenVR.k_unMaxTrackedDeviceCount];
		if (videostreams[deviceIndex] == null)
			videostreams[deviceIndex] = new VideoStream(deviceIndex);
		return videostreams[deviceIndex];
	}

	static VideoStream[] videostreams;

	#endregion
}

