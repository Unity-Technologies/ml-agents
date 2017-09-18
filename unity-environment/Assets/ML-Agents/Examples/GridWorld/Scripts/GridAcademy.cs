using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.UI;
using System.Linq;
using Newtonsoft.Json;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class GridAcademy : Academy
{
    [HideInInspector]
    public List<GameObject> actorObjs;
    [HideInInspector]
    public string[] players;
    [HideInInspector]
    public GameObject trueAgent;
    [HideInInspector]
    public GameObject visualAgent;

    public override void InitializeAcademy()
    {

    }

    public void SetEnvironment()
    {
        trueAgent = GameObject.Find("trueAgent");
        List<string> playersList = new List<string>();
        actorObjs = new List<GameObject>();
        for (int i = 0; i < (int)resetParameters["numObstacles"]; i++)
        {
            playersList.Add("pit");
        }
        playersList.Add("agent");

        for (int i = 0; i < (int)resetParameters["numGoals"]; i++)
        {
            playersList.Add("goal");
        }
        players = playersList.ToArray();
        Camera cam = GameObject.Find("Main Camera").GetComponent<Camera>();

        cam.transform.position = new Vector3(-((int)resetParameters["gridSize"] - 1) / 2f, (int)resetParameters["gridSize"] * 1.25f, -((int)resetParameters["gridSize"] - 1) / 2f);
        cam.orthographicSize = ((int)resetParameters["gridSize"] + 5f) / 2f;
        GameObject.Find("Plane").transform.localScale = new Vector3((int)resetParameters["gridSize"] / 10.0f, 1f, (int)resetParameters["gridSize"] / 10.0f);
        GameObject.Find("Plane").transform.position = new Vector3(((int)resetParameters["gridSize"] - 1) / 2f, -0.5f, ((int)resetParameters["gridSize"] - 1) / 2f);
        GameObject.Find("sN").transform.localScale = new Vector3(1, 1, (int)resetParameters["gridSize"] + 2);
        GameObject.Find("sS").transform.localScale = new Vector3(1, 1, (int)resetParameters["gridSize"] + 2);
        GameObject.Find("sN").transform.position = new Vector3(((int)resetParameters["gridSize"] - 1) / 2f, 0.0f, (int)resetParameters["gridSize"]);
        GameObject.Find("sS").transform.position = new Vector3(((int)resetParameters["gridSize"] - 1) / 2f, 0.0f, -1);
        GameObject.Find("sE").transform.localScale = new Vector3(1, 1, (int)resetParameters["gridSize"] + 2);
        GameObject.Find("sW").transform.localScale = new Vector3(1, 1, (int)resetParameters["gridSize"] + 2);
        GameObject.Find("sE").transform.position = new Vector3((int)resetParameters["gridSize"], 0.0f, ((int)resetParameters["gridSize"] - 1) / 2f);
        GameObject.Find("sW").transform.position = new Vector3(-1, 0.0f, ((int)resetParameters["gridSize"] - 1) / 2f);
        Camera aCam = GameObject.Find("agentCam").GetComponent<Camera>();
        aCam.orthographicSize = ((int)resetParameters["gridSize"]) / 2f;
        aCam.transform.position = new Vector3(((int)resetParameters["gridSize"] - 1) / 2f, (int)resetParameters["gridSize"] + 1f, ((int)resetParameters["gridSize"] - 1) / 2f);

    }

    public override void AcademyReset()
    {
        foreach (GameObject actor in actorObjs)
        {
            DestroyImmediate(actor);
        }
        SetEnvironment();

        actorObjs = new List<GameObject>();

        HashSet<int> numbers = new HashSet<int>();
        while (numbers.Count < players.Length)
        {
            numbers.Add(Random.Range(0, (int)resetParameters["gridSize"] * (int)resetParameters["gridSize"]));
        }
        int[] numbersA = Enumerable.ToArray(numbers);

        for (int i = 0; i < players.Length; i++)
        {
            int x = (numbersA[i]) / (int)resetParameters["gridSize"];
            int y = (numbersA[i]) % (int)resetParameters["gridSize"];
            GameObject actorObj = (GameObject)GameObject.Instantiate(Resources.Load(players[i]));
            actorObj.transform.position = new Vector3(x, -0.25f, y);
            actorObj.name = players[i];
            actorObjs.Add(actorObj);
            if (players[i] == "agent")
            {
                trueAgent.transform.position = actorObj.transform.position;
                trueAgent.transform.rotation = actorObj.transform.rotation;
                visualAgent = actorObj;
            }
        }

    }

    public override void AcademyStep()
    {

    }
}
