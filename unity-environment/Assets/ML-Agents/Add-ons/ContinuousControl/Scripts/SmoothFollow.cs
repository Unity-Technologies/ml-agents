// Smooth Follow from Standard Assets
// If you have C# code and you want to edit SmoothFollow's vars ingame, use this instead.
using System.Collections;
using UnityEngine;

public class SmoothFollow : MonoBehaviour {

	// The target we are following
	public Transform target;
	// The distance in the x-z plane to the target
	public float distance = 10.0f;
	// the height we want the camera to be above the target
	public float height = 5.0f;
	public bool clampToFloor;
	// How much we 
	public float heightDamping = 2.0f;
	public float rotationDamping = 3.0f;

	// Place the script in the Camera-Control group in the component menu
	[AddComponentMenu ("Camera-Control/Smooth Follow")]

	void Start () { }
	void LateUpdate () {
		// Early out if we don't have a target
		if (!target) return;

		// Calculate the current rotation angles
		float wantedRotationAngle = target.eulerAngles.y;
		float wantedHeight = clampToFloor ? height : target.position.y + height;

		float currentRotationAngle = transform.eulerAngles.y;
		float currentHeight = transform.position.y;

		// Damp the rotation around the y-axis
		currentRotationAngle = Mathf.LerpAngle (currentRotationAngle, wantedRotationAngle, rotationDamping * Time.deltaTime);

		// Damp the height
		currentHeight = Mathf.Lerp (currentHeight, wantedHeight, heightDamping * Time.deltaTime);

		// Convert the angle into a rotation
		var currentRotation = Quaternion.Euler (0, currentRotationAngle, 0);

		// Set the position of the camera on the x-z plane to:
		// distance meters behind the target
		transform.position = target.position;
		transform.position -= currentRotation * Vector3.forward * distance;

		// Set the height of the camera
		transform.position = new Vector3 (transform.position.x, currentHeight, transform.position.z);

		// Always look at the target
		transform.LookAt (target);
	}
}