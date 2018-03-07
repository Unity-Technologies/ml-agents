using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentSoccer : Agent
{

    public enum Team
    {
        red, blue
    }
    public enum AgentRole
    {
        striker, defender, goalie
    }
    // ReadRewardData readRewardData;
    public Team team;
    public AgentRole agentRole;
    // public float teamFloat;
    // public float playerID;
    public int playerIndex;
    public SoccerFieldArea area;
    [HideInInspector]
    public Rigidbody agentRB;
    public bool showRaycastRays;
    // [HideInInspector]
    // public Vector3 startingPos;

    public List<float> myState = new List<float>(); //list for state data. to be updated every FixedUpdate in this script
    SoccerAcademy academy;
    Renderer renderer;
    public Vector3 playerDirToTargetGoal;
    public Vector3 playerDirToDefendGoal;

    public float agentEnergy = 100;
    public bool tired = false;
    string[] detectableObjects = { "ball", "wall", "redAgent", "blueAgent" };
    // string[] detectableObjects  = {"ball", "wall", "agent"};

    public void ChooseRandomTeam()
    {
        team = (Team)Random.Range(0, 2);
        renderer.material = team == Team.red ? academy.redMaterial : academy.blueMaterial;
        // area.playerStates[playerIndex].defendGoal = team == Team.red? area.redGoal: area.blueGoal;
    }

    public void JoinRedTeam(AgentRole role)
    {
        agentRole = role;
        team = Team.red;
        // area.playerStates[playerIndex].a
        renderer.material = academy.redMaterial;
    }

    public void JoinBlueTeam(AgentRole role)
    {
        agentRole = role;
        team = Team.blue;
        // area.playerStates[playerIndex].a
        renderer.material = academy.blueMaterial;
    }

    void Awake()
    {
        renderer = GetComponent<Renderer>();
        academy = FindObjectOfType<SoccerAcademy>(); //get the academy
        //brain = agentRole == AgentRole.striker ? academy.brainStriker : agentRole == AgentRole.defender ? academy.brainStriker : academy.brainGoalie;
        PlayerState playerState = new PlayerState();
        playerState.agentRB = GetComponent<Rigidbody>(); //cache the RB
        agentRB = GetComponent<Rigidbody>(); //cache the RB
        agentRB.maxAngularVelocity = 500;
        playerState.startingPos = transform.position;
        playerState.agentScript = this;
        area.playerStates.Add(playerState);
        playerIndex = area.playerStates.IndexOf(playerState);
        playerState.playerIndex = playerIndex;
    }

    public override void InitializeAgent()
    {
        base.InitializeAgent();
    }

    public override void CollectObservations()
    {
        playerDirToTargetGoal = Vector3.zero; //set the target goal based on which team this player is currently on
        playerDirToDefendGoal = Vector3.zero;//set the defend goal based on which team this player is currently on
        Vector3 ballDirToTargetGoal = Vector3.zero;
        Vector3 ballDirToDefendGoal = Vector3.zero;
        if (team == AgentSoccer.Team.red)//I'm on the red team
        {
            playerDirToTargetGoal = area.blueGoal.position - agentRB.position;
            playerDirToDefendGoal = area.redGoal.position - agentRB.position;
            ballDirToTargetGoal = area.blueGoal.position - area.ballRB.position;
            ballDirToDefendGoal = area.redGoal.position - area.ballRB.position;
        }
        if (team == AgentSoccer.Team.blue)//I'm on the blue team
        {
            playerDirToTargetGoal = area.redGoal.position - agentRB.position;
            playerDirToDefendGoal = area.blueGoal.position - agentRB.position;
            ballDirToTargetGoal = area.redGoal.position - area.ballRB.position;
            ballDirToDefendGoal = area.blueGoal.position - area.ballRB.position;
        }

        Vector3 playerPos = agentRB.position - area.ground.transform.position;
        Vector3 ballPos = area.ballRB.position - area.ground.transform.position;
        Vector3 playerDirToBall = area.ballRB.position - agentRB.position;
        AddVectorObs(agentRB.velocity);
        AddVectorObs(playerPos);
        AddVectorObs(playerDirToBall);
        AddVectorObs(playerDirToTargetGoal);
        AddVectorObs(playerDirToDefendGoal);
        AddVectorObs(ballDirToTargetGoal);
        AddVectorObs(ballDirToDefendGoal);
        AddVectorObs(area.ballRB.velocity);

        RaycastAndAddState(agentRB.transform.position, transform.forward); //forward
        RaycastAndAddState(agentRB.transform.position, transform.forward + transform.right); //right forward
        RaycastAndAddState(agentRB.transform.position, transform.right); //right
        RaycastAndAddState(agentRB.transform.position, transform.forward - transform.right); //left forward
        RaycastAndAddState(agentRB.transform.position, -transform.right); //left

        AddVectorObs(agentEnergy / 100);
    }

    public void RaycastAndAddState(Vector3 pos, Vector3 dir)
    {
        RaycastHit hit;
        // float hitDist = 5; //how far away was it. if nothing was hit then this will return our max raycast dist (which is 10 right now)
        // float hitObjHeight = 0;
        // float raycastDist;
        if (showRaycastRays)
        {
            Debug.DrawRay(pos, dir * 30, Color.green, .1f, true);
            // print("drawing debug rays");
        }

        float[] subList = new float[detectableObjects.Length + 5];
        //bit array looks like this
        // [walkableSurface, avoidObstacle, nothing hit, distance] if true 1, else 0
        // [0] walkableSurface
        // [1] walkableSurface
        // [2] no hit
        // [3] hit distance
        var noHitIndex = detectableObjects.Length; //if we didn't hit anything this will be 1
        var hitDistIndex = detectableObjects.Length + 1; //if we hit something the distance will be stored here.
        var hitNormalX = detectableObjects.Length + 2; //if we hit something the distance will be stored here.
        var hitNormalY = detectableObjects.Length + 3; //if we hit something the distance will be stored here.
        var hitNormalZ = detectableObjects.Length + 4; //if we hit something the distance will be stored here.

        // string[] detectableObjects  = { "banana", "agent", "wall", "badBanana", "frozenAgent" };

        // if (Physics.SphereCast(transform.position, 1.0f, position, out hit, rayDistance))
        if (Physics.Raycast(pos, dir, out hit, 30)) // raycast forward to look for walls
                                                    // if (Physics.SphereCast(transform.position, 1.0f, position, out hit, rayDistance))
        {
            for (int i = 0; i < detectableObjects.Length; i++)
            {
                if (hit.collider.gameObject.CompareTag(detectableObjects[i]))
                {
                    subList[i] = 1;  //tag hit
                                     // print("raycast hit: " + detectableObjects[i]);
                    subList[hitDistIndex] = hit.distance / 30; //hit distance is stored in second to last pos

                    subList[hitNormalX] = hit.normal.x;
                    subList[hitNormalY] = hit.normal.y;
                    subList[hitNormalZ] = hit.normal.z;

                    if (team == Team.red && hit.collider.gameObject.CompareTag("redAgent"))
                    {
                        if (hit.distance < 5)
                        {
                            AddReward(-0.001f);
                        }
                    }
                    if (team == Team.blue && hit.collider.gameObject.CompareTag("blueAgent"))
                    {
                        if (hit.distance < 5)
                        {
                            AddReward(-0.001f);
                        }
                    }

                    break;
                }
            }
        }
        else
        {
            subList[noHitIndex] = 1f; //nothing hit
        }
        // stateArray = subList; //for debug
        // print(stateArray);
        AddVectorObs(subList);  //adding n = detectableObjects.Length + 2 items to the state
    }
    public void MoveAgent(float[] act)
    {

        agentEnergy -= Mathf.Abs(act[0]) / 10;
        agentEnergy -= Mathf.Abs(act[1]) / 10;

        Vector3 directionX = Vector3.right * Mathf.Clamp(act[0], -1, 1);  //go left or right in world space
        agentRB.AddForce(directionX * (academy.agentRunSpeed * (agentEnergy / 100) * Random.Range(0, 2)), ForceMode.VelocityChange); //GO
                                                                                                                                     // agentRB.AddForce(directionX * academy.agentRunSpeed, ForceMode.VelocityChange); //GO
                                                                                                                                     // Vector3 directionZ = Vector3.right * Mathf.Clamp(act[1], Random.Range(-1, 0), Random.Range(0,1));  //go left or right in world space
        Vector3 directionZ = Vector3.forward * Mathf.Clamp(act[1], -1, 1); //go forward or back in world space
                                                                           // agentRB.AddForce(directionZ * Random.Range(.3f, 1) * academy.agentRunSpeed, ForceMode.VelocityChange); //GO
        agentRB.AddForce(directionZ * (academy.agentRunSpeed * (agentEnergy / 100) * Random.Range(0, 2)), ForceMode.VelocityChange); //GO
                                                                                                                                     // agentRB.AddForce(directionZ * Random.Range(0, 1) * (academy.agentRunSpeed * Random.Range(0, 2)), ForceMode.VelocityChange); //GO
                                                                                                                                     // Vector3 dirToGo = (directionX * Random.Range(.3f, 1)) + (directionZ * Random.Range(.3f, 1)); //the dir we want to go
        Vector3 dirToGo = directionX + directionZ; //the dir we want to go
                                                   // agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange); //GO
        if (dirToGo != Vector3.zero)
        {
            agentRB.MoveRotation(Quaternion.Lerp(agentRB.rotation, Quaternion.LookRotation(dirToGo), Time.deltaTime * academy.agentRotationSpeed));
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        MoveAgent(vectorAction); //perform agent actions
        if (agentRole == AgentRole.goalie)
        {
            if (playerDirToDefendGoal.sqrMagnitude < 4)
            {
                AddReward(0.001f);  //COACH SAYS: good job
            }
            else
            {
                AddReward(-0.001f); //COACH SAYS: stay by the goal idiot
            }
        }
        float sqrMagnitudeFromAgentToBall = (area.ballRB.position - agentRB.position).sqrMagnitude;

        if (!Physics.Raycast(agentRB.position, Vector3.down, 20)) //if the block has gone over the edge, we done.
        {
            // fail = true; //fell off bro
            AddReward(-1f); // BAD AGENT
                          // ResetBlock(shortBlockRB); //reset block pos
            Done(); //if we mark an agent as done it will be reset automatically. AgentReset() will be called.
        }
    }

    public override void AgentReset()
    {
        transform.position = area.GetRandomSpawnPos();
        agentRB.velocity = Vector3.zero; //we want the agent's vel to return to zero on reset
        if (academy.randomizePlayersTeamForTraining)
        {
            ChooseRandomTeam();
        }
    }

    public override void AgentOnDone()
    {

    }
}
