//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Custom inspector display for SteamVR_Skybox
//
//=============================================================================

using UnityEngine;
using UnityEditor;
using System.Text;
using System.Collections.Generic;
using Valve.VR;
using System.IO;

[CustomEditor(typeof(SteamVR_Skybox)), CanEditMultipleObjects]
public class SteamVR_SkyboxEditor : Editor
{
	private const string nameFormat = "{0}/{1}-{2}.png";
	private const string helpText = "Take snapshot will use the current " +
		"position and rotation to capture six directional screenshots to use as this " +
		"skybox's textures.  Note: This skybox is only used to override what shows up " +
		"in the compositor (e.g. when loading levels).  Add a Camera component to this " +
		"object to override default settings like which layers to render.  Additionally, " +
		"by specifying your own targetTexture, you can control the size of the textures " +
		"and other properties like antialiasing.  Don't forget to disable the camera.\n\n" +
		"For stereo screenshots, a panorama is render for each eye using the specified " +
		"ipd (in millimeters) broken up into segments cellSize pixels square to optimize " +
		"generation.\n(32x32 takes about 10 seconds depending on scene complexity, 16x16 " +
		"takes around a minute, while will 8x8 take several minutes.)\n\nTo test, hit " +
		"play then pause - this will activate the skybox settings, and then drop you to " +
		"the compositor where the skybox is rendered.";

	public override void OnInspectorGUI()
	{
		DrawDefaultInspector();

		EditorGUILayout.HelpBox(helpText, MessageType.Info);

		if (GUILayout.Button("Take snapshot"))
		{
			var directions = new Quaternion[] {
				Quaternion.LookRotation(Vector3.forward),
				Quaternion.LookRotation(Vector3.back),
				Quaternion.LookRotation(Vector3.left),
				Quaternion.LookRotation(Vector3.right),
				Quaternion.LookRotation(Vector3.up, Vector3.back),
				Quaternion.LookRotation(Vector3.down, Vector3.forward)
			};

			Camera tempCamera = null;
			foreach (SteamVR_Skybox target in targets)
			{
				var targetScene = target.gameObject.scene;
                var sceneName = Path.GetFileNameWithoutExtension(targetScene.path);
				var scenePath = Path.GetDirectoryName(targetScene.path);
				var assetPath = scenePath + "/" + sceneName;
				if (!AssetDatabase.IsValidFolder(assetPath))
				{
					var guid = AssetDatabase.CreateFolder(scenePath, sceneName);
					assetPath = AssetDatabase.GUIDToAssetPath(guid);
				}

				var camera = target.GetComponent<Camera>();
				if (camera == null)
				{
					if (tempCamera == null)
						tempCamera = new GameObject().AddComponent<Camera>();
					camera = tempCamera;
				}

				var targetTexture = camera.targetTexture;
				if (camera.targetTexture == null)
				{
					targetTexture = new RenderTexture(1024, 1024, 24);
					targetTexture.antiAliasing = 8;
					camera.targetTexture = targetTexture;
				}

				var oldPosition = target.transform.localPosition;
				var oldRotation = target.transform.localRotation;
				var baseRotation = target.transform.rotation;

				var t = camera.transform;
				t.position = target.transform.position;
				camera.orthographic = false;
				camera.fieldOfView = 90;

				for (int i = 0; i < directions.Length; i++)
				{
					t.rotation = baseRotation * directions[i];
					camera.Render();

					// Copy to texture and save to disk.
					RenderTexture.active = targetTexture;
					var texture = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.ARGB32, false);
					texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
					texture.Apply();
					RenderTexture.active = null;

					var assetName = string.Format(nameFormat, assetPath, target.name, i);
					System.IO.File.WriteAllBytes(assetName, texture.EncodeToPNG());
				}
	
				if (camera != tempCamera)
				{
					target.transform.localPosition = oldPosition;
					target.transform.localRotation = oldRotation;
				}
			}

			if (tempCamera != null)
			{
				Object.DestroyImmediate(tempCamera.gameObject);
			}

			// Now that everything has be written out, reload the associated assets and assign them.
			AssetDatabase.Refresh();
			foreach (SteamVR_Skybox target in targets)
			{
				var targetScene = target.gameObject.scene;
				var sceneName = Path.GetFileNameWithoutExtension(targetScene.path);
				var scenePath = Path.GetDirectoryName(targetScene.path);
				var assetPath = scenePath + "/" + sceneName;

				for (int i = 0; i < directions.Length; i++)
				{
					var assetName = string.Format(nameFormat, assetPath, target.name, i);
					var importer = AssetImporter.GetAtPath(assetName) as TextureImporter;
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					importer.textureFormat = TextureImporterFormat.RGB24;
#else
					importer.textureCompression = TextureImporterCompression.Uncompressed;
#endif
					importer.wrapMode = TextureWrapMode.Clamp;
					importer.mipmapEnabled = false;
					importer.SaveAndReimport();

					var texture = AssetDatabase.LoadAssetAtPath<Texture>(assetName);
					target.SetTextureByIndex(i, texture);
				}
			}
		}
		else if (GUILayout.Button("Take stereo snapshot"))
		{
			const int width = 4096;
			const int height = width / 2;
			const int halfHeight = height / 2;

			var textures = new Texture2D[] {
				new Texture2D(width, height, TextureFormat.ARGB32, false),
				new Texture2D(width, height, TextureFormat.ARGB32, false) };

			var timer = new System.Diagnostics.Stopwatch();

			Camera tempCamera = null;
			foreach (SteamVR_Skybox target in targets)
			{
				timer.Start();

				var targetScene = target.gameObject.scene;
				var sceneName = Path.GetFileNameWithoutExtension(targetScene.path);
				var scenePath = Path.GetDirectoryName(targetScene.path);
				var assetPath = scenePath + "/" + sceneName;
				if (!AssetDatabase.IsValidFolder(assetPath))
				{
					var guid = AssetDatabase.CreateFolder(scenePath, sceneName);
					assetPath = AssetDatabase.GUIDToAssetPath(guid);
				}

				var camera = target.GetComponent<Camera>();
				if (camera == null)
				{
					if (tempCamera == null)
						tempCamera = new GameObject().AddComponent<Camera>();
					camera = tempCamera;
				}

				var fx = camera.gameObject.AddComponent<SteamVR_SphericalProjection>();

				var oldTargetTexture = camera.targetTexture;
				var oldOrthographic = camera.orthographic;
				var oldFieldOfView = camera.fieldOfView;
				var oldAspect = camera.aspect;

				var oldPosition = target.transform.localPosition;
				var oldRotation = target.transform.localRotation;
				var basePosition = target.transform.position;
				var baseRotation = target.transform.rotation;

				var transform = camera.transform;

				int cellSize = int.Parse(target.StereoCellSize.ToString().Substring(1));
	            float ipd = target.StereoIpdMm / 1000.0f;
				int vTotal = halfHeight / cellSize;
				float dv = 90.0f / vTotal; // vertical degrees per segment
				float dvHalf = dv / 2.0f;

				var targetTexture = new RenderTexture(cellSize, cellSize, 24);
				targetTexture.wrapMode = TextureWrapMode.Clamp;
				targetTexture.antiAliasing = 8;

				camera.fieldOfView = dv;
				camera.orthographic = false;
				camera.targetTexture = targetTexture;

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

							var xOffset = -ipd / 2 * Mathf.Cos(pitch * Mathf.Deg2Rad);

							for (int j = 0; j < 2; j++) // left, right
							{
								var texture = textures[j];

								if (j == 1)
								{
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
								texture.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), uTarget, vTarget);
								RenderTexture.active = null;
							}
						}
					}
                }

				// Save textures to disk.
				for (int i = 0; i < 2; i++)
				{
					var texture = textures[i];

					texture.Apply();
					var assetName = string.Format(nameFormat, assetPath, target.name, i);
					File.WriteAllBytes(assetName, texture.EncodeToPNG());
				}

				// Cleanup.
				if (camera != tempCamera)
				{
					camera.targetTexture = oldTargetTexture;
					camera.orthographic = oldOrthographic;
					camera.fieldOfView = oldFieldOfView;
					camera.aspect = oldAspect;

					target.transform.localPosition = oldPosition;
					target.transform.localRotation = oldRotation;
                }
				else
				{
					tempCamera.targetTexture = null;
				}

				DestroyImmediate(targetTexture);
				DestroyImmediate(fx);

				timer.Stop();
				Debug.Log(string.Format("Screenshot took {0} seconds.", timer.Elapsed));
			}

			if (tempCamera != null)
			{
				DestroyImmediate(tempCamera.gameObject);
			}

			DestroyImmediate(textures[0]);
			DestroyImmediate(textures[1]);

			// Now that everything has be written out, reload the associated assets and assign them.
			AssetDatabase.Refresh();
			foreach (SteamVR_Skybox target in targets)
			{
				var targetScene = target.gameObject.scene;
				var sceneName = Path.GetFileNameWithoutExtension(targetScene.path);
				var scenePath = Path.GetDirectoryName(targetScene.path);
				var assetPath = scenePath + "/" + sceneName;

				for (int i = 0; i < 2; i++)
				{
					var assetName = string.Format(nameFormat, assetPath, target.name, i);
					var importer = AssetImporter.GetAtPath(assetName) as TextureImporter;
					importer.mipmapEnabled = false;
					importer.wrapMode = TextureWrapMode.Repeat;
#if (UNITY_5_4 || UNITY_5_3 || UNITY_5_2 || UNITY_5_1 || UNITY_5_0)
					importer.SetPlatformTextureSettings("Standalone", width, TextureImporterFormat.RGB24);
#else
					var settings = importer.GetPlatformTextureSettings("Standalone");
					settings.textureCompression = TextureImporterCompression.Uncompressed;
					settings.maxTextureSize = width;
					importer.SetPlatformTextureSettings(settings);
#endif
					importer.SaveAndReimport();

					var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(assetName);
					target.SetTextureByIndex(i, texture);
				}
			}
		}
	}
}

