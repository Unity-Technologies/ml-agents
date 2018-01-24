//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Utilities for working with SteamVR
//
//=============================================================================

using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using Valve.VR;

public static class SteamVR_Utils
{
	// this version does not clamp [0..1]
	public static Quaternion Slerp(Quaternion A, Quaternion B, float t)
	{
		var cosom = Mathf.Clamp(A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w, -1.0f, 1.0f);
		if (cosom < 0.0f)
		{
			B = new Quaternion(-B.x, -B.y, -B.z, -B.w);
			cosom = -cosom;
		}

		float sclp, sclq;
		if ((1.0f - cosom) > 0.0001f)
		{
			var omega = Mathf.Acos(cosom);
			var sinom = Mathf.Sin(omega);
			sclp = Mathf.Sin((1.0f - t) * omega) / sinom;
			sclq = Mathf.Sin(t * omega) / sinom;
		}
		else
		{
			// "from" and "to" very close, so do linear interp
			sclp = 1.0f - t;
			sclq = t;
		}

		return new Quaternion(
			sclp * A.x + sclq * B.x,
			sclp * A.y + sclq * B.y,
			sclp * A.z + sclq * B.z,
			sclp * A.w + sclq * B.w);
	}

	public static Vector3 Lerp(Vector3 A, Vector3 B, float t)
	{
		return new Vector3(
			Lerp(A.x, B.x, t),
			Lerp(A.y, B.y, t),
			Lerp(A.z, B.z, t));
	}

	public static float Lerp(float A, float B, float t)
	{
		return A + (B - A) * t;
	}

	public static double Lerp(double A, double B, double t)
	{
		return A + (B - A) * t;
	}

	public static float InverseLerp(Vector3 A, Vector3 B, Vector3 result)
	{
		return Vector3.Dot(result - A, B - A);
	}

	public static float InverseLerp(float A, float B, float result)
	{
		return (result - A) / (B - A);
	}

	public static double InverseLerp(double A, double B, double result)
	{
		return (result - A) / (B - A);
	}

	public static float Saturate(float A)
	{
		return (A < 0) ? 0 : (A > 1) ? 1 : A;
	}

	public static Vector2 Saturate(Vector2 A)
	{
		return new Vector2(Saturate(A.x), Saturate(A.y));
	}

	public static float Abs(float A)
	{
		return (A < 0) ? -A : A;
	}

	public static Vector2 Abs(Vector2 A)
	{
		return new Vector2(Abs(A.x), Abs(A.y));
	}

	private static float _copysign(float sizeval, float signval)
	{
		return Mathf.Sign(signval) == 1 ? Mathf.Abs(sizeval) : -Mathf.Abs(sizeval);
	}

	public static Quaternion GetRotation(this Matrix4x4 matrix)
	{
		Quaternion q = new Quaternion();
		q.w = Mathf.Sqrt(Mathf.Max(0, 1 + matrix.m00 + matrix.m11 + matrix.m22)) / 2;
		q.x = Mathf.Sqrt(Mathf.Max(0, 1 + matrix.m00 - matrix.m11 - matrix.m22)) / 2;
		q.y = Mathf.Sqrt(Mathf.Max(0, 1 - matrix.m00 + matrix.m11 - matrix.m22)) / 2;
		q.z = Mathf.Sqrt(Mathf.Max(0, 1 - matrix.m00 - matrix.m11 + matrix.m22)) / 2;
		q.x = _copysign(q.x, matrix.m21 - matrix.m12);
		q.y = _copysign(q.y, matrix.m02 - matrix.m20);
		q.z = _copysign(q.z, matrix.m10 - matrix.m01);
		return q;
	}

	public static Vector3 GetPosition(this Matrix4x4 matrix)
	{
		var x = matrix.m03;
		var y = matrix.m13;
		var z = matrix.m23;

		return new Vector3(x, y, z);
	}

	public static Vector3 GetScale(this Matrix4x4 m)
	{
		var x = Mathf.Sqrt(m.m00 * m.m00 + m.m01 * m.m01 + m.m02 * m.m02);
		var y = Mathf.Sqrt(m.m10 * m.m10 + m.m11 * m.m11 + m.m12 * m.m12);
		var z = Mathf.Sqrt(m.m20 * m.m20 + m.m21 * m.m21 + m.m22 * m.m22);

		return new Vector3(x, y, z);
	}

	[System.Serializable]
	public struct RigidTransform
	{
		public Vector3 pos;
		public Quaternion rot;

		public static RigidTransform identity
		{
			get { return new RigidTransform(Vector3.zero, Quaternion.identity); }
		}

		public static RigidTransform FromLocal(Transform t)
		{
			return new RigidTransform(t.localPosition, t.localRotation);
		}

		public RigidTransform(Vector3 pos, Quaternion rot)
		{
			this.pos = pos;
			this.rot = rot;
		}

		public RigidTransform(Transform t)
		{
			this.pos = t.position;
			this.rot = t.rotation;
		}

		public RigidTransform(Transform from, Transform to)
		{
			var inv = Quaternion.Inverse(from.rotation);
			rot = inv * to.rotation;
			pos = inv * (to.position - from.position);
		}

		public RigidTransform(HmdMatrix34_t pose)
		{
			var m = Matrix4x4.identity;

			m[0, 0] =  pose.m0;
			m[0, 1] =  pose.m1;
			m[0, 2] = -pose.m2;
			m[0, 3] =  pose.m3;

			m[1, 0] =  pose.m4;
			m[1, 1] =  pose.m5;
			m[1, 2] = -pose.m6;
			m[1, 3] =  pose.m7;

			m[2, 0] = -pose.m8;
			m[2, 1] = -pose.m9;
			m[2, 2] =  pose.m10;
			m[2, 3] = -pose.m11;

			this.pos = m.GetPosition();
			this.rot = m.GetRotation();
		}

		public RigidTransform(HmdMatrix44_t pose)
		{
			var m = Matrix4x4.identity;

			m[0, 0] =  pose.m0;
			m[0, 1] =  pose.m1;
			m[0, 2] = -pose.m2;
			m[0, 3] =  pose.m3;

			m[1, 0] =  pose.m4;
			m[1, 1] =  pose.m5;
			m[1, 2] = -pose.m6;
			m[1, 3] =  pose.m7;

			m[2, 0] = -pose.m8;
			m[2, 1] = -pose.m9;
			m[2, 2] =  pose.m10;
			m[2, 3] = -pose.m11;

			m[3, 0] =  pose.m12;
			m[3, 1] =  pose.m13;
			m[3, 2] = -pose.m14;
			m[3, 3] =  pose.m15;

			this.pos = m.GetPosition();
			this.rot = m.GetRotation();
		}

		public HmdMatrix44_t ToHmdMatrix44()
		{
			var m = Matrix4x4.TRS(pos, rot, Vector3.one);
			var pose = new HmdMatrix44_t();

			pose.m0  =  m[0, 0];
            pose.m1  =  m[0, 1];
			pose.m2  = -m[0, 2];
			pose.m3  =  m[0, 3];

			pose.m4  =  m[1, 0];
			pose.m5  =  m[1, 1];
			pose.m6  = -m[1, 2];
			pose.m7  =  m[1, 3];

			pose.m8  = -m[2, 0];
			pose.m9  = -m[2, 1];
			pose.m10 =  m[2, 2];
			pose.m11 = -m[2, 3];

			pose.m12 =  m[3, 0];
			pose.m13 =  m[3, 1];
			pose.m14 = -m[3, 2];
			pose.m15 =  m[3, 3];

			return pose;
		}

		public HmdMatrix34_t ToHmdMatrix34()
		{
			var m = Matrix4x4.TRS(pos, rot, Vector3.one);
			var pose = new HmdMatrix34_t();

			pose.m0  =  m[0, 0];
            pose.m1  =  m[0, 1];
			pose.m2  = -m[0, 2];
			pose.m3  =  m[0, 3];

			pose.m4  =  m[1, 0];
			pose.m5  =  m[1, 1];
			pose.m6  = -m[1, 2];
			pose.m7  =  m[1, 3];

			pose.m8  = -m[2, 0];
			pose.m9  = -m[2, 1];
			pose.m10 =  m[2, 2];
			pose.m11 = -m[2, 3];

			return pose;
		}

		public override bool Equals(object o)
		{
			if (o is RigidTransform)
			{
				RigidTransform t = (RigidTransform)o;
				return pos == t.pos && rot == t.rot;
			}
			return false;
		}

		public override int GetHashCode()
		{
			return pos.GetHashCode() ^ rot.GetHashCode();
		}

		public static bool operator ==(RigidTransform a, RigidTransform b)
		{
			return a.pos == b.pos && a.rot == b.rot;
		}

		public static bool operator !=(RigidTransform a, RigidTransform b)
		{
			return a.pos != b.pos || a.rot != b.rot;
		}

		public static RigidTransform operator *(RigidTransform a, RigidTransform b)
		{
			return new RigidTransform
			{
				rot = a.rot * b.rot,
				pos = a.pos + a.rot * b.pos
			};
		}

		public void Inverse()
		{
			rot = Quaternion.Inverse(rot);
			pos = -(rot * pos);
		}

		public RigidTransform GetInverse()
		{
			var t = new RigidTransform(pos, rot);
			t.Inverse();
			return t;
		}

		public void Multiply(RigidTransform a, RigidTransform b)
		{
			rot = a.rot * b.rot;
			pos = a.pos + a.rot * b.pos;
		}

		public Vector3 InverseTransformPoint(Vector3 point)
		{
			return Quaternion.Inverse(rot) * (point - pos);
		}

		public Vector3 TransformPoint(Vector3 point)
		{
			return pos + (rot * point);
		}

		public static Vector3 operator *(RigidTransform t, Vector3 v)
		{
			return t.TransformPoint(v);
		}

		public static RigidTransform Interpolate(RigidTransform a, RigidTransform b, float t)
		{
			return new RigidTransform(Vector3.Lerp(a.pos, b.pos, t), Quaternion.Slerp(a.rot, b.rot, t));
		}

		public void Interpolate(RigidTransform to, float t)
		{
			pos = SteamVR_Utils.Lerp(pos, to.pos, t);
			rot = SteamVR_Utils.Slerp(rot, to.rot, t);
		}
	}

	public delegate object SystemFn(CVRSystem system, params object[] args);

	public static object CallSystemFn(SystemFn fn, params object[] args)
	{
		var initOpenVR = (!SteamVR.active && !SteamVR.usingNativeSupport);
		if (initOpenVR)
		{
			var error = EVRInitError.None;
			OpenVR.Init(ref error, EVRApplicationType.VRApplication_Utility);
		}

		var system = OpenVR.System;
		var result = (system != null) ? fn(system, args) : null;

		if (initOpenVR)
			OpenVR.Shutdown();

		return result;
	}

	public static void TakeStereoScreenshot(uint screenshotHandle, GameObject target, int cellSize, float ipd, ref string previewFilename, ref string VRFilename)
	{
		const int width = 4096;
		const int height = width / 2;
		const int halfHeight = height / 2;

		var texture = new Texture2D(width, height * 2, TextureFormat.ARGB32, false);

		var timer = new System.Diagnostics.Stopwatch();

		Camera tempCamera = null;

		timer.Start();

		var camera = target.GetComponent<Camera>();
		if (camera == null)
		{
			if (tempCamera == null)
				tempCamera = new GameObject().AddComponent<Camera>();
			camera = tempCamera;
		}

		// Render preview texture
		const int previewWidth = 2048;
		const int previewHeight = 2048;
		var previewTexture = new Texture2D(previewWidth, previewHeight, TextureFormat.ARGB32, false);
		var targetPreviewTexture = new RenderTexture(previewWidth, previewHeight, 24);

		var oldTargetTexture = camera.targetTexture;
		var oldOrthographic = camera.orthographic;
		var oldFieldOfView = camera.fieldOfView;
		var oldAspect = camera.aspect;
		var oldstereoTargetEye = camera.stereoTargetEye;
		camera.stereoTargetEye = StereoTargetEyeMask.None;
		camera.fieldOfView = 60.0f;
		camera.orthographic = false;
		camera.targetTexture = targetPreviewTexture;
		camera.aspect = 1.0f;
		camera.Render();

		// copy preview texture
		RenderTexture.active = targetPreviewTexture;
		previewTexture.ReadPixels(new Rect(0, 0, targetPreviewTexture.width, targetPreviewTexture.height), 0, 0);
		RenderTexture.active = null;
		camera.targetTexture = null;
		Object.DestroyImmediate(targetPreviewTexture);

		var fx = camera.gameObject.AddComponent<SteamVR_SphericalProjection>();

		var oldPosition = target.transform.localPosition;
		var oldRotation = target.transform.localRotation;
		var basePosition = target.transform.position;
		var baseRotation = Quaternion.Euler(0, target.transform.rotation.eulerAngles.y, 0);

		var transform = camera.transform;

		int vTotal = halfHeight / cellSize;
		float dv = 90.0f / vTotal; // vertical degrees per segment
		float dvHalf = dv / 2.0f;

		var targetTexture = new RenderTexture(cellSize, cellSize, 24);
		targetTexture.wrapMode = TextureWrapMode.Clamp;
		targetTexture.antiAliasing = 8;

		camera.fieldOfView = dv;
		camera.orthographic = false;
		camera.targetTexture = targetTexture;
		camera.aspect = oldAspect;
		camera.stereoTargetEye = StereoTargetEyeMask.None;

		// Render sections of a sphere using a rectilinear projection
		// and resample using a sphereical projection into a single panorama
		// texture per eye.  We break into sections in order to keep the eye
		// separation similar around the sphere.  Rendering alternates between
		// top and bottom sections, sweeping horizontally around the sphere,
		// alternating left and right eyes.
		for (int v = 0; v < vTotal; v++)
		{
			var pitch = 90.0f - (v * dv) - dvHalf;
			var uTotal = width / targetTexture.width;
			var du = 360.0f / uTotal; // horizontal degrees per segment
			var duHalf = du / 2.0f;

			var vTarget = v * halfHeight / vTotal;

			for (int i = 0; i < 2; i++) // top, bottom
			{
				if (i == 1)
				{
					pitch = -pitch;
					vTarget = height - vTarget - cellSize;
				}

				for (int u = 0; u < uTotal; u++)
				{
					var yaw = -180.0f + (u * du) + duHalf;

					var uTarget = u * width / uTotal;

					var vTargetOffset = 0;
					var xOffset = -ipd / 2 * Mathf.Cos(pitch * Mathf.Deg2Rad);

					for (int j = 0; j < 2; j++) // left, right
					{
						if (j == 1)
						{
							vTargetOffset = height;
							xOffset = -xOffset;
						}

						var offset = baseRotation * Quaternion.Euler(0, yaw, 0) * new Vector3(xOffset, 0, 0);
						transform.position = basePosition + offset;

						var direction = Quaternion.Euler(pitch, yaw, 0.0f);
						transform.rotation = baseRotation * direction;

						// vector pointing to center of this section
						var N = direction * Vector3.forward;

						// horizontal span of this section in degrees
						var phi0 = yaw - (du / 2);
						var phi1 = phi0 + du;

						// vertical span of this section in degrees
						var theta0 = pitch + (dv / 2);
						var theta1 = theta0 - dv;

						var midPhi = (phi0 + phi1) / 2;
						var baseTheta = Mathf.Abs(theta0) < Mathf.Abs(theta1) ? theta0 : theta1;

						// vectors pointing to corners of image closes to the equator
						var V00 = Quaternion.Euler(baseTheta, phi0, 0.0f) * Vector3.forward;
						var V01 = Quaternion.Euler(baseTheta, phi1, 0.0f) * Vector3.forward;

						// vectors pointing to top and bottom midsection of image
						var V0M = Quaternion.Euler(theta0, midPhi, 0.0f) * Vector3.forward;
						var V1M = Quaternion.Euler(theta1, midPhi, 0.0f) * Vector3.forward;

						// intersection points for each of the above
						var P00 = V00 / Vector3.Dot(V00, N);
						var P01 = V01 / Vector3.Dot(V01, N);
						var P0M = V0M / Vector3.Dot(V0M, N);
						var P1M = V1M / Vector3.Dot(V1M, N);

						// calculate basis vectors for plane
						var P00_P01 = P01 - P00;
						var P0M_P1M = P1M - P0M;

						var uMag = P00_P01.magnitude;
						var vMag = P0M_P1M.magnitude;

						var uScale = 1.0f / uMag;
						var vScale = 1.0f / vMag;

						var uAxis = P00_P01 * uScale;
						var vAxis = P0M_P1M * vScale;

						// update material constant buffer
						fx.Set(N, phi0, phi1, theta0, theta1,
							uAxis, P00, uScale,
							vAxis, P0M, vScale);

						camera.aspect = uMag / vMag;
						camera.Render();

						RenderTexture.active = targetTexture;
						texture.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), uTarget, vTarget + vTargetOffset);
						RenderTexture.active = null;                 
					}

					// Update progress
					var progress = (float)( v * ( uTotal * 2.0f ) + u + i*uTotal) / (float)(vTotal * ( uTotal * 2.0f ) );
					OpenVR.Screenshots.UpdateScreenshotProgress(screenshotHandle, progress);
				}
			}
		}

		// 100% flush
		OpenVR.Screenshots.UpdateScreenshotProgress(screenshotHandle, 1.0f);

		// Save textures to disk.
		// Add extensions
		previewFilename += ".png";
		VRFilename += ".png";

		// Preview
		previewTexture.Apply();
		System.IO.File.WriteAllBytes(previewFilename, previewTexture.EncodeToPNG());

		// VR
		texture.Apply();
		System.IO.File.WriteAllBytes(VRFilename, texture.EncodeToPNG());

		// Cleanup.
		if (camera != tempCamera)
		{
			camera.targetTexture = oldTargetTexture;
			camera.orthographic = oldOrthographic;
			camera.fieldOfView = oldFieldOfView;
			camera.aspect = oldAspect;
			camera.stereoTargetEye = oldstereoTargetEye;

			target.transform.localPosition = oldPosition;
			target.transform.localRotation = oldRotation;
		}
		else
		{
			tempCamera.targetTexture = null;
		}

		Object.DestroyImmediate(targetTexture);
		Object.DestroyImmediate(fx);

		timer.Stop();
		Debug.Log(string.Format("Screenshot took {0} seconds.", timer.Elapsed));

		if (tempCamera != null)
		{
			Object.DestroyImmediate(tempCamera.gameObject);
		}

		Object.DestroyImmediate(previewTexture);
		Object.DestroyImmediate(texture);
	}
}

