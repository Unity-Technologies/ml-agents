using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraZoom : MonoBehaviour {

	Camera camera;

	void Awake()
	{
		camera = Camera.main;
	}
	public void SetCameraHeight(float height)
	{
		camera.transform.position = new Vector3(camera.transform.position.x, height, camera.transform.position.z);
	}
}
