using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HumanoidAcademy : Academy {

	public GameObject humanoidPrefab;
	public float gravityMultiplier;
	public float strength;
	public Brain brain;
	public Transform spawnPoint1;
	public Transform spawnPoint2;
	public Transform spawnPoint3;
	public Transform spawnPoint4;
	public Transform spawnPoint5;
	public Transform spawnPoint6;
	public Transform spawnPoint7;
	public Transform spawnPoint8;
	public Transform spawnPoint9;
	public Transform spawnPoint10;
	public float standingStrength;
	CameraFollow cameraFollow;
	public bool resetPositionOnFail;
	public float angularVelTarget;
	public float maxAngularVelocity;
	// public Transform spawnPoint1;
	// Use this for initialization
	void Start () {
		SpawnPlayer(spawnPoint1);
		SpawnPlayer(spawnPoint2);
		SpawnPlayer(spawnPoint3);
		SpawnPlayer(spawnPoint4);
		SpawnPlayer(spawnPoint5);
		SpawnPlayer(spawnPoint6);
		SpawnPlayer(spawnPoint7);
		SpawnPlayer(spawnPoint8);
		SpawnPlayer(spawnPoint9);
		SpawnPlayer(spawnPoint10);
		cameraFollow = FindObjectOfType<CameraFollow>();
		// cameraFollow.target = 
	}
	
	public void SpawnPlayer(Transform spawnPoint)
	{

		// print("spawn player");
		// GameObject player = Instantiate(humanoidPrefab, new Vector3(0f,2.043f, 0f), Quaternion.identity);
		GameObject player = Instantiate(humanoidPrefab, new Vector3(spawnPoint.position.x, 2.043f, spawnPoint.position.z), Quaternion.identity);
		HumanoidAgent agentScript = player.GetComponent<HumanoidAgent>();
		agentScript.spawnPoint = spawnPoint;
		agentScript.GiveBrain(brain);
    	// agentScript.AgentReset();
		// player.GetComponent<HumanoidAgent>().brain = brain;

	}
	public void SpawnPlayerAndDestroyThisOne(Transform spawnPoint, HumanoidAgent agentToDestroy)
	{
		// DestroyImmediate(oldPlayer);
		// Destroy(oldPlayer);
		// oldPlayer.SetActive(false);
		// print("instantiate new player");
		DestroyImmediate(agentToDestroy.gameObject);
		GameObject player = Instantiate(humanoidPrefab, new Vector3(spawnPoint.position.x, 2.043f, spawnPoint.position.z), Quaternion.identity);
		HumanoidAgent agentScript = player.GetComponent<HumanoidAgent>();
		agentScript.spawnPoint = spawnPoint;
		agentScript.GiveBrain(brain);
    	// agentScript.AgentReset();
		// done = true;

		// player.GetComponent<HumanoidAgent>().brain = brain;

	}

	public override void InitializeAcademy()
	{
		// SpawnPlayer();
		// base.InitializeAcademy();
	}
	public override void AcademyReset()
	{

	}
	// Update is called once per frame
	void Update () {
		if(cameraFollow.target == null)
		{
			cameraFollow.target = FindObjectOfType<HumanoidAgent>().hips.transform;
		}
		
	}





}
