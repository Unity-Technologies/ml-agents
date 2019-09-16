using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents;


public class GridAcademy : Academy
{
    [HideInInspector]
    public List<GameObject> actorObjs;
    [HideInInspector]
    public int[] players;

    public GameObject trueAgent;

    public int gridSize;

    public GameObject camObject;
    Camera m_Cam;
    Camera m_AgentCam;

    public GameObject agentPref;
    public GameObject goalPref;
    public GameObject pitPref;
    GameObject[] m_Objects;

    GameObject m_Plane;
    GameObject m_Sn;
    GameObject m_Ss;
    GameObject m_Se;
    GameObject m_Sw;

    public override void InitializeAcademy()
    {
        gridSize = (int)resetParameters["gridSize"];
        m_Cam = camObject.GetComponent<Camera>();

        m_Objects = new[] {agentPref, goalPref, pitPref};

        m_AgentCam = GameObject.Find("agentCam").GetComponent<Camera>();

        actorObjs = new List<GameObject>();

        m_Plane = GameObject.Find("Plane");
        m_Sn = GameObject.Find("sN");
        m_Ss = GameObject.Find("sS");
        m_Sw = GameObject.Find("sW");
        m_Se = GameObject.Find("sE");
    }

    public void SetEnvironment()
    {
        m_Cam.transform.position = new Vector3(-((int)resetParameters["gridSize"] - 1) / 2f,
            (int)resetParameters["gridSize"] * 1.25f,
            -((int)resetParameters["gridSize"] - 1) / 2f);
        m_Cam.orthographicSize = ((int)resetParameters["gridSize"] + 5f) / 2f;

        var playersList = new List<int>();

        for (var i = 0; i < (int)resetParameters["numObstacles"]; i++)
        {
            playersList.Add(2);
        }

        for (var i = 0; i < (int)resetParameters["numGoals"]; i++)
        {
            playersList.Add(1);
        }
        players = playersList.ToArray();

        m_Plane.transform.localScale = new Vector3(gridSize / 10.0f, 1f, gridSize / 10.0f);
        m_Plane.transform.position = new Vector3((gridSize - 1) / 2f, -0.5f, (gridSize - 1) / 2f);
        m_Sn.transform.localScale = new Vector3(1, 1, gridSize + 2);
        m_Ss.transform.localScale = new Vector3(1, 1, gridSize + 2);
        m_Sn.transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, gridSize);
        m_Ss.transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, -1);
        m_Se.transform.localScale = new Vector3(1, 1, gridSize + 2);
        m_Sw.transform.localScale = new Vector3(1, 1, gridSize + 2);
        m_Se.transform.position = new Vector3(gridSize, 0.0f, (gridSize - 1) / 2f);
        m_Sw.transform.position = new Vector3(-1, 0.0f, (gridSize - 1) / 2f);

        m_AgentCam.orthographicSize = (gridSize) / 2f;
        m_AgentCam.transform.position = new Vector3((gridSize - 1) / 2f, gridSize + 1f, (gridSize - 1) / 2f);
    }

    public override void AcademyReset()
    {
        foreach (var actor in actorObjs)
        {
            DestroyImmediate(actor);
        }
        SetEnvironment();

        actorObjs.Clear();

        var numbers = new HashSet<int>();
        while (numbers.Count < players.Length + 1)
        {
            numbers.Add(Random.Range(0, gridSize * gridSize));
        }
        var numbersA = Enumerable.ToArray(numbers);

        for (var i = 0; i < players.Length; i++)
        {
            var x = (numbersA[i]) / gridSize;
            var y = (numbersA[i]) % gridSize;
            var actorObj = Instantiate(m_Objects[players[i]]);
            actorObj.transform.position = new Vector3(x, -0.25f, y);
            actorObjs.Add(actorObj);
        }

        var xA = (numbersA[players.Length]) / gridSize;
        var yA = (numbersA[players.Length]) % gridSize;
        trueAgent.transform.position = new Vector3(xA, -0.25f, yA);
    }

    public override void AcademyStep()
    {
    }
}
