using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HallwayAcademy : Academy {

	public float agentRunSpeed; 
	public float agentRotationSpeed;
    public Material goalScoredMaterial; //when a goal is scored the ground will use this material for a few seconds.
    public Material failMaterial; //when fail, the ground will use this material for a few seconds. 
	public float gravityMultiplier; //use ~3 to make things less floaty

	public override void AcademyReset()
	{

    }
	// Use this for initialization
	void Start () {
		Physics.gravity *= gravityMultiplier; //Normally you shouldn't override Start() or Awake() with MLAgents, but Start() isn't used in Academy.cs so this should be ok for now.
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
