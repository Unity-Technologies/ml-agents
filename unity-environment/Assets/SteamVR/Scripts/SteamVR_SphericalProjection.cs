//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Applies spherical projection to output.
//
//=============================================================================

using UnityEngine;

[ExecuteInEditMode]
public class SteamVR_SphericalProjection : MonoBehaviour
{
	static Material material;

	public void Set(Vector3 N,
		float phi0, float phi1, float theta0, float theta1, // in degrees
		Vector3 uAxis, Vector3 uOrigin, float uScale,
		Vector3 vAxis, Vector3 vOrigin, float vScale)
	{
		if (material == null)
			material = new Material(Shader.Find("Custom/SteamVR_SphericalProjection"));

		material.SetVector("_N", new Vector4(N.x, N.y, N.z));
		material.SetFloat("_Phi0", phi0 * Mathf.Deg2Rad);
		material.SetFloat("_Phi1", phi1 * Mathf.Deg2Rad);
		material.SetFloat("_Theta0", theta0 * Mathf.Deg2Rad + Mathf.PI / 2);
		material.SetFloat("_Theta1", theta1 * Mathf.Deg2Rad + Mathf.PI / 2);
		material.SetVector("_UAxis", uAxis);
		material.SetVector("_VAxis", vAxis);
		material.SetVector("_UOrigin", uOrigin);
		material.SetVector("_VOrigin", vOrigin);
		material.SetFloat("_UScale", uScale);
		material.SetFloat("_VScale", vScale);
	}

	void OnRenderImage(RenderTexture src, RenderTexture dest)
	{
		Graphics.Blit(src, dest, material);
	}
}

