//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Simple two bone ik solver.
//
//=============================================================================

using UnityEngine;

public class SteamVR_IK : MonoBehaviour
{
	public Transform target;
	public Transform start, joint, end;
	public Transform poleVector, upVector;

	public float blendPct = 1.0f;

	[HideInInspector]
	public Transform startXform, jointXform, endXform;

	void LateUpdate()
	{
		const float epsilon = 0.001f;
		if (blendPct < epsilon)
			return;

		var preUp = upVector ? upVector.up : Vector3.Cross(end.position - start.position, joint.position - start.position).normalized;

		var targetPosition = target.position;
		var targetRotation = target.rotation;

		Vector3 forward, up, result = joint.position;
		Solve(start.position, targetPosition, poleVector.position,
			(joint.position - start.position).magnitude,
			(end.position - joint.position).magnitude,
			ref result, out forward, out up);

		if (up == Vector3.zero)
			return;

		var startPosition = start.position;
		var jointPosition = joint.position;
		var endPosition = end.position;

		var startRotationLocal = start.localRotation;
		var jointRotationLocal = joint.localRotation;
		var endRotationLocal = end.localRotation;

		var startParent = start.parent;
		var jointParent = joint.parent;
		var endParent = end.parent;

		var startScale = start.localScale;
		var jointScale = joint.localScale;
		var endScale = end.localScale;

		if (startXform == null)
		{
			startXform = new GameObject("startXform").transform;
			startXform.parent = transform;
		}

		startXform.position = startPosition;
		startXform.LookAt(joint, preUp);
		start.parent = startXform;

		if (jointXform == null)
		{
			jointXform = new GameObject("jointXform").transform;
			jointXform.parent = startXform;
		}

		jointXform.position = jointPosition;
		jointXform.LookAt(end, preUp);
		joint.parent = jointXform;

		if (endXform == null)
		{
			endXform = new GameObject("endXform").transform;
			endXform.parent = jointXform;
		}

		endXform.position = endPosition;
		end.parent = endXform;

		startXform.LookAt(result, up);
		jointXform.LookAt(targetPosition, up);
		endXform.rotation = targetRotation;

		start.parent = startParent;
		joint.parent = jointParent;
		end.parent = endParent;

		end.rotation = targetRotation; // optionally blend?

		// handle blending in/out
		if (blendPct < 1.0f)
		{
			start.localRotation = Quaternion.Slerp(startRotationLocal, start.localRotation, blendPct);
			joint.localRotation = Quaternion.Slerp(jointRotationLocal, joint.localRotation, blendPct);
			end.localRotation = Quaternion.Slerp(endRotationLocal, end.localRotation, blendPct);
		}

		// restore scale so it doesn't blow out
		start.localScale = startScale;
		joint.localScale = jointScale;
		end.localScale = endScale;
	}

	public static bool Solve(
		Vector3 start, // shoulder / hip
		Vector3 end, // desired hand / foot position
		Vector3 poleVector, // point to aim elbow / knee toward
		float jointDist, // distance from start to elbow / knee
		float targetDist, // distance from joint to hand / ankle
		ref Vector3 result, // original and output elbow / knee position
		out Vector3 forward, out Vector3 up) // plane formed by root, joint and target
	{
		var totalDist = jointDist + targetDist;
		var start2end = end - start;
		var poleVectorDir = (poleVector - start).normalized;
		var baseDist = start2end.magnitude;

		result = start;

		const float epsilon = 0.001f;
		if (baseDist < epsilon)
		{
			// move jointDist toward jointTarget
			result += poleVectorDir * jointDist;

			forward = Vector3.Cross(poleVectorDir, Vector3.up);
			up = Vector3.Cross(forward, poleVectorDir).normalized;
		}
		else
		{
			forward = start2end * (1.0f / baseDist);
			up = Vector3.Cross(forward, poleVectorDir).normalized;

			if (baseDist + epsilon < totalDist)
			{
				// calculate the area of the triangle to determine its height
				var p = (totalDist + baseDist) * 0.5f; // half perimeter
				if (p > jointDist + epsilon && p > targetDist + epsilon)
				{
					var A = Mathf.Sqrt(p * (p - jointDist) * (p - targetDist) * (p - baseDist));
					var height = 2.0f * A / baseDist; // distance of joint from line between root and target

					var dist = Mathf.Sqrt((jointDist * jointDist) - (height * height));
					var right = Vector3.Cross(up, forward); // no need to normalized - already orthonormal

					result += (forward * dist) + (right * height);
					return true; // in range
				}
				else
				{
					// move jointDist toward jointTarget
					result += poleVectorDir * jointDist;
				}
			}
			else
			{
				// move elboDist toward target
				result += forward * jointDist;
			}
		}

		return false; // edge cases
	}
}

