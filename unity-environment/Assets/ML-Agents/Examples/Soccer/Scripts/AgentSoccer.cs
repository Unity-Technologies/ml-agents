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
    float kickPower;
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

    public void RayPerception(float rayDistance,
                             float[] rayAngles, string[] detectableObjects, float startHeight, float endHeight)
    {
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(GiveCatersian(rayDistance, noisyAngle));
            position.y = endHeight;
            Debug.DrawRay(transform.position + new Vector3(0f, endHeight, 0f), position, Color.red, 0.1f, true);
            RaycastHit hit;
            float[] subList = new float[detectableObjects.Length + 2];
            if (Physics.SphereCast(transform.position + new Vector3(0f, endHeight, 0f), 1.0f, position, out hit, rayDistance))
            {
                for (int i = 0; i < detectableObjects.Length; i++)
                {
                    if (hit.collider.gameObject.CompareTag(detectableObjects[i]))
                    {
                        subList[i] = 1;
                        subList[detectableObjects.Length + 1] = hit.distance / rayDistance;
                        break;
                    }
                }
            }
            else
            {
                subList[detectableObjects.Length] = 1f;
            }
            foreach (float f in subList)
                AddVectorObs(f);
        }
    }

    public Vector3 GiveCatersian(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian(angle));
        float z = radius * Mathf.Sin(DegreeToRadian(angle));
        return new Vector3(x, 1f, z);
    }

    public float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }

    public override void CollectObservations()
    {
        float rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        string[] detectableObjects;
        if (team == Team.red)
        {
            detectableObjects = new string[] { "ball", "redGoal", "blueGoal", "wall" };
        }
        else
        {
            detectableObjects = new string[] { "ball", "blueGoal", "redGoal", "wall" };
        }
        RayPerception(rayDistance, rayAngles, detectableObjects, 0f, 0f);
        RayPerception(rayDistance, rayAngles, detectableObjects, 1f, 1f);
    }

    public void MoveAgent(float[] act)
    {
        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;


        // If we're using Continuous control you will need to change the Action
        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            //// Larger the value, the less the penalty is.
            float energyConservPenaltyModifier = 10000;

            //// The larger the movement, the greater the penalty given.
            AddReward(-Mathf.Abs(act[0]) / energyConservPenaltyModifier);
            AddReward(-Mathf.Abs(act[1]) / energyConservPenaltyModifier);

            dirToGo = transform.forward * Mathf.Clamp(act[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(act[1], -1f, 1f);
            kickPower = Mathf.Clamp(act[2], 0f, 1f);
        }
        else
        {
            kickPower = 0f;
            int action = Mathf.FloorToInt(act[0]);
            if (action == 0)
            {
                dirToGo = transform.forward * 1f;
                kickPower = 1f;
            }
            else if (action == 1)
            {
                dirToGo = transform.forward * -1f;
            }
            else if (action == 2)
            {
                rotateDir = transform.up * 1f;
            }
            else if (action == 3)
            {
                rotateDir = transform.up * -1f;
            }
            else if (action == 4)
            {
                dirToGo = transform.right * -1f;
            }
            else if (action == 5)
            {
                dirToGo = transform.right * 1f;
            }
        }
        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange); // GO

    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (agentRole == AgentRole.striker)
        {
            AddReward(-1f / 3000f);
        }
        if (agentRole == AgentRole.goalie)
        {
            AddReward(1f / 3000f);
        }
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

    void OnCollisionEnter(Collision c)
    {
        // force is how forcefully we will push the player away from the enemy.
        float force = 2000f * kickPower;

        // If the object we hit is the enemy
        if (c.gameObject.tag == "ball")
        {
            // Calculate Angle Between the collision point and the player
            Vector3 dir = c.contacts[0].point - transform.position;
            // We then get the opposite (-Vector3) and normalize it
            dir = dir.normalized;
            // And finally we add force in the direction of dir and multiply it by force. 
            // This will push back the player
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }


    public override void AgentReset()
    {
        if (academy.randomizePlayersTeamForTraining)
        {
            ChooseRandomTeam();
        }

        if (team == Team.red)
        {
            transform.rotation = Quaternion.Euler(0f, -90f + Random.Range(-10f, 10f), 0f);
        }
        else
        {
            transform.rotation = Quaternion.Euler(0f, 90f + Random.Range(-10f, 10f), 0f);
        }
        transform.position = area.GetRandomSpawnPos(team.ToString(), agentRole.ToString());
        agentRB.velocity = Vector3.zero; //we want the agent's vel to return to zero on reset
        agentRB.angularVelocity = Vector3.zero;
        area.ResetBall();
    }

    public override void AgentOnDone()
    {

    }
}
