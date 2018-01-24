//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Test SteamVR_Controller support.
//
//=============================================================================

using UnityEngine;
using System.Collections.Generic;
using Valve.VR;

public class SteamVR_TestController : MonoBehaviour
{
	List<int> controllerIndices = new List<int>();

	private void OnDeviceConnected(int index, bool connected)
	{
		var system = OpenVR.System;
		if (system == null || system.GetTrackedDeviceClass((uint)index) != ETrackedDeviceClass.Controller)
			return;

		if (connected)
		{
			Debug.Log(string.Format("Controller {0} connected.", index));
			PrintControllerStatus(index);
			controllerIndices.Add(index);
		}
		else
		{
			Debug.Log(string.Format("Controller {0} disconnected.", index));
			PrintControllerStatus(index);
			controllerIndices.Remove(index);
		}
	}

	void OnEnable()
	{
		SteamVR_Events.DeviceConnected.Listen(OnDeviceConnected);
	}

	void OnDisable()
	{
		SteamVR_Events.DeviceConnected.Remove(OnDeviceConnected);
	}

	void PrintControllerStatus(int index)
	{
		var device = SteamVR_Controller.Input(index);
		Debug.Log("index: " + device.index);
		Debug.Log("connected: " + device.connected);
		Debug.Log("hasTracking: " + device.hasTracking);
		Debug.Log("outOfRange: " + device.outOfRange);
		Debug.Log("calibrating: " + device.calibrating);
		Debug.Log("uninitialized: " + device.uninitialized);
		Debug.Log("pos: " + device.transform.pos);
		Debug.Log("rot: " + device.transform.rot.eulerAngles);
		Debug.Log("velocity: " + device.velocity);
		Debug.Log("angularVelocity: " + device.angularVelocity);

		var l = SteamVR_Controller.GetDeviceIndex(SteamVR_Controller.DeviceRelation.Leftmost);
		var r = SteamVR_Controller.GetDeviceIndex(SteamVR_Controller.DeviceRelation.Rightmost);
		Debug.Log((l == r) ? "first" : (l == index) ? "left" : "right");
	}

	EVRButtonId[] buttonIds = new EVRButtonId[] {
		EVRButtonId.k_EButton_ApplicationMenu,
		EVRButtonId.k_EButton_Grip,
		EVRButtonId.k_EButton_SteamVR_Touchpad,
		EVRButtonId.k_EButton_SteamVR_Trigger
	};

	EVRButtonId[] axisIds = new EVRButtonId[] {
		EVRButtonId.k_EButton_SteamVR_Touchpad,
		EVRButtonId.k_EButton_SteamVR_Trigger
	};

	public Transform point, pointer;

	void Update()
	{
		foreach (var index in controllerIndices)
		{
			var overlay = SteamVR_Overlay.instance;
			if (overlay && point && pointer)
			{
				var t = SteamVR_Controller.Input(index).transform;
				pointer.transform.localPosition = t.pos;
				pointer.transform.localRotation = t.rot;

				var results = new SteamVR_Overlay.IntersectionResults();
				var hit = overlay.ComputeIntersection(t.pos, t.rot * Vector3.forward, ref results);
				if (hit)
				{
					point.transform.localPosition = results.point;
					point.transform.localRotation = Quaternion.LookRotation(results.normal);
				}

				continue;
			}

			foreach (var buttonId in buttonIds)
			{
				if (SteamVR_Controller.Input(index).GetPressDown(buttonId))
					Debug.Log(buttonId + " press down");
				if (SteamVR_Controller.Input(index).GetPressUp(buttonId))
				{
					Debug.Log(buttonId + " press up");
					if (buttonId == EVRButtonId.k_EButton_SteamVR_Trigger)
					{
						SteamVR_Controller.Input(index).TriggerHapticPulse();
						PrintControllerStatus(index);
					}
				}
				if (SteamVR_Controller.Input(index).GetPress(buttonId))
					Debug.Log(buttonId);
			}

			foreach (var buttonId in axisIds)
			{
				if (SteamVR_Controller.Input(index).GetTouchDown(buttonId))
					Debug.Log(buttonId + " touch down");
				if (SteamVR_Controller.Input(index).GetTouchUp(buttonId))
					Debug.Log(buttonId + " touch up");
				if (SteamVR_Controller.Input(index).GetTouch(buttonId))
				{
					var axis = SteamVR_Controller.Input(index).GetAxis(buttonId);
					Debug.Log("axis: " + axis);
				}
			}
		}
	}
}

