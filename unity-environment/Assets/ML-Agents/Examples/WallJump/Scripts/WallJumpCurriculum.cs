using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallJumpCurriculum : MonoBehaviour {

	WallJumpAcademy academy;

	// public bool lesson1;
	// public bool lesson2;
	// public bool lesson3;
	// public bool lesson4;

	// public int currentLesson;

	// // Use this for initialization
	void Start () {
		academy = FindObjectOfType<WallJumpAcademy>();
	}
	
	// // Update is called once per frame
	// void Update () {
	// 	if(Input.GetKeyDown(KeyCode.Alpha1))
	// 	{
	// 		AllLessonsFalse();
	// 		currentLesson = 1;

	// 		// GoToLesson1();
	// 	}
		
	// }



	// void AllLessonsFalse()
	// {
	// 	lesson1 = false;
	// 	lesson2 = false;
	// 	lesson3 = false;
	// 	lesson4 = false;
	// }

	public void SetWallHeight(float height)
	{
		academy.wallHeight = height;
		// camera.transform.position = new Vector3(camera.transform.position.x, height, camera.transform.position.z);
	}
}
