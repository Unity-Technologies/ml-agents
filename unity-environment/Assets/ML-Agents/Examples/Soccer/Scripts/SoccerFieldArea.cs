using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public class PlayerState
{
    public int playerIndex; //index pos on the team
    public List<float> state = new List<float>(); //list for state data. to be updated every FixedUpdate in this script
    public Rigidbody agentRB; //the agent's rb
    public Vector3 startingPos; //the agent's starting position
    public AgentSoccer agentScript; //this is the agent's script
    public float ballPosReward;
}

public class SoccerFieldArea : MonoBehaviour
{


    public Transform redGoal;
    public Transform blueGoal;

    public AgentSoccer redStriker;
    public AgentSoccer blueStriker;
    public AgentSoccer redGoalie;
    public AgentSoccer blueGoalie;
    public GameObject ball;
    [HideInInspector]
    public Rigidbody ballRB;
    public GameObject ground; //to be used to determine spawn areas
    public GameObject centerPitch; //to be used to determine spawn areas
    SoccerBallController ballController;
    public List<PlayerState> playerStates = new List<PlayerState>();
    [HideInInspector]
    public Vector3 ballStartingPos;
    Bounds areaBounds;
    public bool drawSpawnAreaGizmo;
    Vector3 spawnAreaSize;
    public float goalScoreByTeamReward; //if red scores they get this reward & vice versa
    public float goalScoreAgainstTeamReward; //if red scores we deduct some from blue & vice versa
    public GameObject goalTextUI;
    public float totalPlayers;
    [HideInInspector]
    public bool canResetBall;
    public bool useSpawnPoint;
    public Transform spawnPoint;
    Material groundMaterial;
    Renderer groundRenderer;
    SoccerAcademy academy;
    public float blueBallPosReward;
    public float redBallPosReward;



    public IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        groundRenderer.material = mat;
        yield return new WaitForSeconds(time); //wait for 2 sec
        groundRenderer.material = groundMaterial;
    }


    void Awake()
    {
        academy = FindObjectOfType<SoccerAcademy>();
        groundRenderer = centerPitch.GetComponent<Renderer>(); //get the ground renderer so we can change the material when a goal is scored
        groundMaterial = groundRenderer.material; //starting material
        canResetBall = true;
        if (goalTextUI) { goalTextUI.SetActive(false); }
        ballRB = ball.GetComponent<Rigidbody>();
        ballController = ball.GetComponent<SoccerBallController>();
        ballController.area = this;
        ballStartingPos = ball.transform.position;
        Mesh mesh = ground.GetComponent<MeshFilter>().mesh;  //get the ground's mesh
        areaBounds = ground.GetComponent<Collider>().bounds; //get the ground's bounds
    }

    IEnumerator ShowGoalUI()
    {
        if (goalTextUI) goalTextUI.SetActive(true);
        yield return new WaitForSeconds(.25f);
        if (goalTextUI) goalTextUI.SetActive(false);
    }

    public void AllPlayersDone(float reward)
    {
        foreach (PlayerState ps in playerStates)
        {
            // if(ps.agentScript.gameObject.activeInHierarchy &&  ps.agentScript.agentRole == AgentSoccer.AgentRole.striker || ps.agentScript.agentRole == AgentSoccer.AgentRole.defender)
            if (ps.agentScript.gameObject.activeInHierarchy)
            {
                if (reward != 0)
                {
                    ps.agentScript.AddReward(reward);
                }
                ps.agentScript.Done();
            }

        }
    }




    // public void GoalScored(string goalTag)
    // {

    // }
    public void BlueGoalTouched() //ball touched the blue goal
    {
        if (academy.randomizeFieldOrientationForTraining)
        {
            // print("rotating field");
            ground.transform.Rotate(Vector3.up * Random.Range(-10, 10));
            // this.transform.rotation *= Quaternion.Euler(0, Random.Range(0, 360), 0); //rotate by a random amount on the y axis.
        }
        foreach (PlayerState ps in playerStates)
        {
            if (ps.agentScript.team == AgentSoccer.Team.blue) //if currently on the blue team you suck
            {
                RewardOrPunishPlayer(ps, academy.strikerPunish, academy.defenderPunish, academy.goaliePunish);
            }
            else if (ps.agentScript.team == AgentSoccer.Team.red) //if currently on the red team you get a reward

            // if(ps.agentScript.team == AgentSoccer.Team.red) //if currently on the red team you get a reward
            {
                RewardOrPunishPlayer(ps, academy.strikerReward, academy.defenderReward, academy.goalieReward);
            }

            if (academy.randomizePlayersTeamForTraining)
            {
                ps.agentScript.ChooseRandomTeam();
                // ps.currentTeamFloat = Random.Range(0,2); //return either a 0 or 1 * max is exclusive ex: Random.Range(0,10) will pick a int between 0-9
            }
        }



        StartCoroutine(GoalScoredSwapGroundMaterial(academy.redMaterial, 2));
        //ResetBall();
        if (goalTextUI)
        {
            StartCoroutine(ShowGoalUI());
        }
    }


    public void RedGoalTouched() //ball touched the blue goal
    {
        if (academy.randomizeFieldOrientationForTraining)
        {
            // print("rotating field");

            ground.transform.Rotate(Vector3.up * Random.Range(-10, 10));

            // transform.Rotate(new Vector3(0, Random.Range(0, 360), 0), Space.Self);
            // this.transform.rotation *= Quaternion.Euler(0, Random.Range(0, 360), 0); //rotate by a random amount on the y axis.
        }
        foreach (PlayerState ps in playerStates)
        {
            if (ps.agentScript.team == AgentSoccer.Team.blue) //if currently on the blue team you get a reward
            {
                RewardOrPunishPlayer(ps, academy.strikerReward, academy.defenderReward, academy.goalieReward);
            }
            else if (ps.agentScript.team == AgentSoccer.Team.red) //if currently on the red team you suck
            {
                RewardOrPunishPlayer(ps, academy.strikerPunish, academy.defenderPunish, academy.goaliePunish);
            }
            if (academy.randomizePlayersTeamForTraining)
            {
                ps.agentScript.ChooseRandomTeam();
                // ps.currentTeamFloat = Random.Range(0,2); //return either a 0 or 1 * max is exclusive ex: Random.Range(0,10) will pick a int between 0-9
            }
        }

        StartCoroutine(GoalScoredSwapGroundMaterial(academy.blueMaterial, 2));
        //ResetBall();
        if (goalTextUI)
        {
            StartCoroutine(ShowGoalUI());
        }
    }

    public void RewardOrPunishPlayer(PlayerState ps, float striker, float defender, float goalie) //ball touched the red goal
    {
        if (ps.agentScript.agentRole == AgentSoccer.AgentRole.striker)
        {
            ps.agentScript.AddReward(striker);
        }
        if (ps.agentScript.agentRole == AgentSoccer.AgentRole.defender)
        {
            ps.agentScript.AddReward(defender);
        }
        if (ps.agentScript.agentRole == AgentSoccer.AgentRole.goalie)
        {
            ps.agentScript.AddReward(goalie);
        }
        ps.agentScript.Done();  //all agents need to be reset
    }


    //DEBUG AREA BOUNDS
    void OnDrawGizmos()
    {
        if (drawSpawnAreaGizmo)
        {
            // spawnAreaSize = areaBounds.size * spawnAreaMarginMultiplier;
            spawnAreaSize = areaBounds.extents;
            Gizmos.DrawWireCube(ground.transform.position, spawnAreaSize);
        }
    }

    public Vector3 GetRandomSpawnPos(string type, string position)
    {
        Vector3 randomSpawnPos = Vector3.zero;
        float xOffset = 0f;
        if (type == "red")
        {
            if (position == "goalie")
            {
                xOffset = 13f;
            }
            if (position == "striker")
            {
                xOffset = 7f;
            }
        }
        if (type == "blue")
        {
            if (position == "goalie")
            {
                xOffset = -13f;
            }
            if (position == "striker")
            {
                xOffset = -7f;
            }
        }
        randomSpawnPos = ground.transform.position + new Vector3(xOffset, 0f, 0f) + (Random.insideUnitSphere * 2);
        randomSpawnPos.y = ground.transform.position.y + 1;
        return randomSpawnPos;
    }

    void SpawnObjAtPos(GameObject obj, Vector3 pos)
    {
        obj.transform.position = pos;
        // canUseThisPos = true;
    }

    public void ResetBall()
    {
        ball.transform.position = GetRandomSpawnPos("ball", "ball");
        ballRB.velocity = Vector3.zero;
        ballRB.angularVelocity = Vector3.zero;
    }
}
