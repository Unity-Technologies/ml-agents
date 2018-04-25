using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HumanoidAgent : Agent {
  	// [HideInInspector]
    public bool fell;
 	// public float strength;


	Vector3 past_velocity;
	// Brain agentBrain;
	public HumanoidAcademy academy;
	// public Rigidbody[] bodyPartsNoHandsNoFeet;
	public Rigidbody[] allBodyParts;
	public List<Rigidbody> moveableBodyParts = new List<Rigidbody>();
	public Rigidbody hips;
	public Rigidbody spine;
	public Rigidbody chest;
	public Rigidbody head;
	public float targetHeightHips;
	public float targetHeightChest;
	public int bodyPartsCount;
	public int moveableBodyPartsCount;
	[HideInInspector]
	public Transform spawnPoint;

	void Awake()
	{
		academy = FindObjectOfType<HumanoidAcademy>();
		brain = academy.brain;
		targetHeightChest = chest.position.y;
		targetHeightHips = hips.position.y;
		// Rigidbody[] bodyPartsTemp = GetComponentsInChildren<Rigidbody>(true);
		allBodyParts = GetComponentsInChildren<Rigidbody>(true);
		// foreach(Rigidbody rb in bodyPartsTemp)
		// {
		// 	if(rb != hips)
		// 	{
		// 		bodyParts.Add(rb);
		// 	}
		// }
		foreach(Rigidbody rb in allBodyParts)
		{
			// ConfigurableJoint joint = rb.gameObject.GetComponent<ConfigurableJoint>();
			// if (joint)
			// {
			// 	JointDrive jd = new JointDrive();
			// 	jd .positionSpring =10;
			// 	jd.positionDamper = 0;
			// 	jd .maximumForce =1000;
			// 	joint.slerpDrive = jd;
			// }
			rb.maxAngularVelocity = 30;
			// rb.mass = 1;
			rb.drag = .5f;
			rb.angularDrag = .5f;
			// rb.maxAngularVelocity = 500;
			if(rb.name != "neck" && rb.name != "foot_L" && rb.transform.name != "foot_R" && rb.name != "hand_L" && rb.name != "hand_R" && rb.name != "shoulder_L" && rb.name != "shoulder_R" )
			{
				moveableBodyParts.Add(rb);
			}
		}
		// agentBrain = FindObjectOfType<Brain>();
		bodyPartsCount = allBodyParts.Length;
		moveableBodyPartsCount = moveableBodyParts.Count;
		// bodyPartsCount = bodyParts.Count;
	}
    public override void InitializeAgent()
    {
		base.InitializeAgent();
	}
	public override void AgentReset()
    {
		// print("Done. spawn a new player");
		// academy.SpawnPlayer();
		// academy.SpawnPlayerAndDestroyThisOne(spawnPoint, this);
		// if(academy.resetPositionOnFail)
		// {
		// 	academy.SpawnPlayerAndDestroyThisOne(spawnPoint, this);

		// }
    }


    public override void CollectObservations()
    // public override List<float> CollectState()
    {
			// MLAgentsHelpers.CollectVector3State(state, hips.transform.position);
			// MLAgentsHelpers.CollectVector3State(state, hips.velocity);
			// MLAgentsHelpers.CollectVector3State(state, hips.angularVelocity);
			// MLAgentsHelpers.CollectRotationState(state, hips.transform);
		// foreach(Rigidbody rb in allBodyParts)
		foreach(Rigidbody rb in moveableBodyParts)
		{
			// ConfigurableJoint joint = rb.gameObject.GetComponent<ConfigurableJoint>();
			// if (joint)
			// {
			// 	print("current force: " + joint.currentForce.sqrMagnitude);
			// 	print("current torque: " + joint.currentTorque.sqrMagnitude);
			// 	// JointDrive jd = new JointDrive();
			// 	// jd .positionSpring =10;
			// 	// jd.positionDamper = 0;
			// 	// jd .maximumForce =1000;
			// 	// joint.slerpDrive = jd;
			// }
			// if(rb)
			// MLAgentsHelpers.CollectVector3State(state, rb.transform.position);
			// AddVectorObs(rb.transform.localPosition);

			// AddVectorObs(hips.transform.localPosition);
			AddVectorObs(rb.transform.position - hips.position);
			AddVectorObs(rb.velocity);
			AddVectorObs(rb.angularVelocity);
			AddVectorObs(rb.transform.localRotation);
			// MLAgentsHelpers.CollectVector3State(state, rb.transform.localPosition);
			// MLAgentsHelpers.CollectVector3State(state, rb.velocity);
			// MLAgentsHelpers.CollectVector3State(state, rb.angularVelocity);
			// MLAgentsHelpers.CollectLocalRotationState(state, rb.transform);
			// MLAgentsHelpers.CollectRotationState(state, rb.transform);
		}
			AddVectorObs(hips.transform.localPosition);
			AddVectorObs(hips.transform.localRotation);
			AddVectorObs(hips.position.y);
			AddVectorObs(head.position.y);
			AddVectorObs(chest.position.y);
			// MLAgentsHelpers.CollectVector3State(state, hips.transform.localPosition);
			// MLAgentsHelpers.CollectVector3State(state, hips.velocity);
			// MLAgentsHelpers.CollectVector3State(state, hips.angularVelocity);
			// MLAgentsHelpers.CollectRotationState(state, hips.transform);


		// return state;

	}

    // public override void CollectObservations()
    // // public override List<float> CollectState()
    // {
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.transform.position);
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.velocity);
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.angularVelocity);
	// 		// MLAgentsHelpers.CollectRotationState(state, hips.transform);
	// 	// foreach(Rigidbody rb in allBodyParts)
	// 	foreach(Rigidbody rb in moveableBodyParts)
	// 	{
	// 		// if(rb)
	// 		// MLAgentsHelpers.CollectVector3State(state, rb.transform.position);
	// 		// AddVectorObs(rb.transform.localPosition);
	// 		AddVectorObs(hips.position - rb.transform.position);
	// 		AddVectorObs(rb.velocity);
	// 		AddVectorObs(rb.angularVelocity);
	// 		AddVectorObs(rb.transform.localRotation);
	// 		// MLAgentsHelpers.CollectVector3State(state, rb.transform.localPosition);
	// 		// MLAgentsHelpers.CollectVector3State(state, rb.velocity);
	// 		// MLAgentsHelpers.CollectVector3State(state, rb.angularVelocity);
	// 		// MLAgentsHelpers.CollectLocalRotationState(state, rb.transform);
	// 		// MLAgentsHelpers.CollectRotationState(state, rb.transform);
	// 	}
	// 		AddVectorObs(hips.rotation);
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.transform.localPosition);
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.velocity);
	// 		// MLAgentsHelpers.CollectVector3State(state, hips.angularVelocity);
	// 		// MLAgentsHelpers.CollectRotationState(state, hips.transform);


	// 	// return state;

	// }

	// void MoveVelocityTowards(Vector3 targetPos, Rigidbody rb, AnimationCurve curve, float curveTimer, float targetVel, float maxVel)
	// void MoveVelocityTowards(Vector3 targetPos, Rigidbody rb)
	// {
	// 	Vector3 moveToPos = targetPos - rb.worldCenterOfMass;  //cube needs to go to the standard Pos
	// 	Vector3 velocityTarget = (moveToPos * targetVel * curve.Evaluate(curveTimer)) * Time.deltaTime; //not sure of the logic here, but it modifies velTarget
	// 	if (float.IsNaN(velocityTarget.x) == false)
	// 	{
	// 		rb.velocity = Vector3.MoveTowards(rb.velocity, velocityTarget, maxVel);
	// 	}
	// }

	void MoveAngularVelocityTowards(Vector3 targetPos, Rigidbody rb)
	{
		Vector3 moveToPos = targetPos - rb.worldCenterOfMass;  //target pos
		Vector3 moveVector = (moveToPos * academy.angularVelTarget) * Time.deltaTime; 
		if (float.IsNaN(moveVector.x) == false) //sanity check
		{
			rb.angularVelocity = Vector3.MoveTowards(rb.angularVelocity, moveVector, academy.maxAngularVelocity);
		}
	}

	void MoveAgent(float[] vectorAction)
	{
		// int actionLength = vectorAction.Length;
		// int bodyPartsArrayLength = bodyParts.Length;
		// print(bodyPartsArrayLength + " ");
		// print(actionLength + " ");
		// int currentActionIndex = 0;


		for(int x = 0; x < moveableBodyPartsCount - 1; x++)
		{
			var rb = moveableBodyParts[x];
			var torqueRight = 		rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 3], -1, 1);
			var torqueUp = 			rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1);
			var torqueForward = 		rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 2], -1, 1);
			// var torqueDirRight = 		rb.transform.right * 	Mathf.Clamp(vectorAction[x * 6], -1, 1);
			// var torqueDirUp = 			rb.transform.up * 		Mathf.Clamp(vectorAction[(x * 6) + 1], -1, 1);
			// var torqueDirForward = 		rb.transform.forward *	Mathf.Clamp(vectorAction[(x * 6) + 2], -1, 1);
			// var torqueStrengthRight = 	academy.strength * 		Mathf.Clamp(vectorAction[(x * 6) + 3], 0, 1);
			// var torqueStrengthUp = 		academy.strength *		Mathf.Clamp(vectorAction[(x * 6) + 4], 0, 1);
			// var torqueStrengthForward = academy.strength * 		Mathf.Clamp(vectorAction[(x * 6) + 5], 0, 1);

			// Vector3 torqueRight = Vector3.Lerp(startMarker.position, endMarker.position, fracJourney)
			// Vector3 angVelocityRight = rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 3], -1, 1);
			// Vector3 angVelocityUp = rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1);
			// Vector3 angVelocityForward = rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 2], -1, 1);

			// Quaternion deltaRotationRight = Quaternion.Euler(angVelocityRight * Time.deltaTime);
			// Quaternion deltaRotationUp= Quaternion.Euler(angVelocityUp * Time.deltaTime);
			// Quaternion deltaRotationForward = Quaternion.Euler(angVelocityForward * Time.deltaTime);
			Quaternion deltaRotationRight = Quaternion.Euler(torqueRight * Time.deltaTime);
			Quaternion deltaRotationUp= Quaternion.Euler(torqueUp * Time.deltaTime);
			Quaternion deltaRotationForward = Quaternion.Euler(torqueForward * Time.deltaTime);
			rb.MoveRotation(rb.rotation * deltaRotationRight);
			rb.MoveRotation(rb.rotation * deltaRotationUp);
			rb.MoveRotation(rb.rotation * deltaRotationForward);





				// rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 3], -1, 1), ForceMode.Force);
				// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.Force);
				// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 2], -1, 1), ForceMode.Force);
					// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// reward -= (Mathf.Abs(act[x * 3] + act[(x * 3) + 1] +  act[(x * 3) + 2]))/1000;
				



				// AddReward(-((Mathf.Abs(vectorAction[x * 3]) + Mathf.Abs(vectorAction[(x * 3) + 1]) + Mathf.Abs(vectorAction[(x * 3) + 2]))/100));
				float torque_penalty = (Mathf.Abs(vectorAction[x * 3]) * Mathf.Abs(vectorAction[x * 3]))
				+ (Mathf.Abs(vectorAction[(x * 3) + 1]) * Mathf.Abs(vectorAction[(x * 3) + 1]))
				+ (Mathf.Abs(vectorAction[(x * 3) + 2]) * Mathf.Abs(vectorAction[(x * 3) + 2]));


				AddReward(-(torque_penalty/100));
				
				
				// }
				// else
				// {
				// 	rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.VelocityChange);
				// 	// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.VelocityChange);
				// 	rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// }

				// rb.AddTorque(rb.transform.right * academy.strength * vectorAction[x * 2], ForceMode.VelocityChange);
				// rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1));
				// rb.AddTorque(rb.transform.forward * academy.strength * vectorAction[(x * 2) + 1], ForceMode.VelocityChange);
				// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 1], -1, 1));
				// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1));
				// rb.AddTorque(rb.transform.right * academy.strength * vectorAction[x * 3]);
				// rb.AddTorque(rb.transform.up * academy.strength * vectorAction[(x * 3) + 1]);
				// rb.AddTorque(rb.transform.forward * academy.strength * vectorAction[(x * 3) + 2]);
				// reward -= (Mathf.Abs(act[x * 3] + act[(x * 3) + 1] +  act[(x * 3) + 2]))/1000;
				// AddReward(-(rb.angularVelocity.sqrMagnitude/1000));


				// if(rb.name == "hips" || rb.name == "spine" || rb.name == "chest")
				// {
				// 	// rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.Impulse);
				// 	// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.Impulse);
				// 	rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.VelocityChange);
				// 	rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.VelocityChange);
				// 	// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// }
				// else
				// {
				// 	rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.VelocityChange);
				// 	// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.VelocityChange);
				// 	rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// }
			}
			// if(rb.name != "head")
			// {
				
			// 	// if(rb.name == "hips" || rb.name == "spine" || rb.name == "chest"|| rb.name == "neck" )
				// if(rb.name == "hips" || rb.name == "spine" || rb.name == "chest")
				// {
				// 	// rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.Impulse);
				// 	// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.Impulse);
				// 	rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.VelocityChange);
				// 	rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.VelocityChange);
				// 	// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// }
				// else
				// {
				// 	rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1), ForceMode.VelocityChange);
				// 	// rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1), ForceMode.VelocityChange);
				// 	rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1), ForceMode.VelocityChange);
				// }

			// 	// rb.AddTorque(rb.transform.right * academy.strength * vectorAction[x * 2], ForceMode.VelocityChange);
			// 	// rb.AddTorque(rb.transform.right * academy.strength * Mathf.Clamp(vectorAction[x * 2], -1, 1));
			// 	// rb.AddTorque(rb.transform.forward * academy.strength * vectorAction[(x * 2) + 1], ForceMode.VelocityChange);
			// 	// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 1], -1, 1));
			// 	// rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 2) + 2], -1, 1));
			// 	// rb.AddTorque(rb.transform.right * academy.strength * vectorAction[x * 3]);
			// 	// rb.AddTorque(rb.transform.up * academy.strength * vectorAction[(x * 3) + 1]);
			// 	// rb.AddTorque(rb.transform.forward * academy.strength * vectorAction[(x * 3) + 2]);
			// 	// reward -= (Mathf.Abs(act[x * 3] + act[(x * 3) + 1] +  act[(x * 3) + 2]))/1000;
			// }
			// else
			// {
			// 	print("head");
			// }
				
		// }


		// for(int x = 0; x <  - 1; x++)
		// {
		// 	var rb = allBodyParts[x];
		// 	rb.AddTorque(rb.transfbodyPartsCountorm.right * academy.strength * Mathf.Clamp(vectorAction[x * 3], -1, 1));
		// 	rb.AddTorque(rb.transform.up * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 1], -1, 1));
		// 	rb.AddTorque(rb.transform.forward * academy.strength * Mathf.Clamp(vectorAction[(x * 3) + 2], -1, 1));
		// 	// rb.AddTorque(rb.transform.right * academy.strength * vectorAction[x * 3]);
		// 	// rb.AddTorque(rb.transform.up * academy.strength * vectorAction[(x * 3) + 1]);
		// 	// rb.AddTorque(rb.transform.forward * academy.strength * vectorAction[(x * 3) + 2]);
		// 	// reward -= (Mathf.Abs(act[x * 3] + act[(x * 3) + 1] +  act[(x * 3) + 2]))/1000;
		// }

		// hips.AddForce(Vector3.up * academy.standingStrength * Mathf.Clamp(vectorAction[28], 0, 1), ForceMode.Acceleration);
		// chest.AddForce(Vector3.up * academy.standingStrength * Mathf.Clamp(vectorAction[28], 0, 1));

		// hips.AddTorque(hips.transform.right * academy.strength * act[0], ForceMode.VelocityChange);
		// hips.AddTorque(hips.transform.up * academy.strength * act[1], ForceMode.VelocityChange);
		// hips.AddTorque(hips.transform.forward * academy.strength * act[2], ForceMode.VelocityChange);
		// foreach(Rigidbody rb in bodyParts)
		// {
		// 	rb.AddTorque(rb.transform.right * academy.strength * act)
		// }
	}


	/// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
	public override void AgentAction(float[] vectorAction, string textAction)
    {
    // public override void AgentStep(float[] act)
    // {
		// print("agent step");
		MoveAgent(vectorAction);
		// reward -= chest.velocity.sqrMagnitude/10000;
		// if (!done)
        // {
			// AddReward(head.position.y/100);
			// AddReward(chest.position.y/100);
			// AddReward(hips.position.y/100);

			// reward += chest.position.y/100;
			// reward += hips.position.y/100;
			// AddReward(chest.position.y/100);
			// AddReward(hips.position.y/100);


		// if(chest.position.y < targetHeightChest - 1)
		// {
		// 	// done = true;
		// 	AddReward(-1);
		// 	Done();
		// 	// reward = -1;
		// }
		// else
		// {

		// 	// AddReward(chest.position.y/100);
		// 	AddReward(head.position.y/100);
		// 	// AddReward(hips.position.y/100);
		// }
		if(head.position.y < .35f)
		{
			AddReward(-1);
			Done();
		}
		else
		{
			AddReward(head.position.y/10);
			// AddReward(chest.position.y/100);
			// AddReward(hips.position.y/100);
		}


			// reward += targetHeightChest - chest.position.y;
			// reward += targetHeightHips - hips.position.y;
			// if(targetHeightChest - chest.position.y < .1f)
			// {
			// 	reward += .1f;
			// }
			// if(targetHeightHips - hips.position.y < .1f)
			// {
			// 	reward += .1f;
			// }

            // reward = (0
            // // - 0.01f * torque_penalty
            // + 1.0f * hips.GetComponent<Rigidbody>().velocity.x
            // // - 0.05f * Mathf.Abs(body.transform.position.z - body.transform.parent.transform.position.z)
            // // - 0.05f * Mathf.Abs(body.GetComponent<Rigidbody>().velocity.y)
            // );
        // }
		if (fell)
		{
			// print("fell");
			AddReward(-1);
			// done = true;
			// reward = -1;
			fell = false;
			Done();
		}

	}

	public override void AgentOnDone()
	{
		// academy.SpawnPlayer();
		if(academy.resetPositionOnFail)
		{
			academy.SpawnPlayerAndDestroyThisOne(spawnPoint, this);

		}

		// Destroy(gameObject);
	}

	// void ResetPosition()



}
