using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallJumpAcademy : Academy {

	public float agentRunSpeed; 
	public float agentRotationSpeed;
	public float spawnAreaMarginMultiplier; //ex: .9 means 90% of spawn area will be used.... .1 margin will be left (so players don't spawn off of the edge). the higher this value, the longer training time required
    public Material goalScoredMaterial; //when a goal is scored the ground will use this material for a few seconds.
    public Material failMaterial; //when fail, the ground will use this material for a few seconds. 

	public float gravityMultiplier; //use ~3 to make things less floaty
	public float currentWallHeight;
	public float agentJumpHeight;
	public float agentJumpVelocity;
	public float agentJumpVelocityMaxChange;
	public float agentRaycastDistance;
	// public float minWallHeight;
    // public float maxWallHeight;
	public float wallHeight;

	public override void AcademyReset()
	{
		// wallHeight = 
        //wallHeight = (float)resetParameters["wall_height"];
        // minWallHeight = (float)resetParameters["min_wall_height"];
        // maxWallHeight = (float)resetParameters["max_wall_height"];
	}
	// Use this for initialization
    public override void InitializeAcademy()
    {
		Physics.gravity *= gravityMultiplier; //Normally you shouldn't override Start() or Awake() with MLAgents, but Start() isn't used in Academy.cs so this should be ok for now.
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
