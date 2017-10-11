using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongViewer : MonoBehaviour {
    public PongAcademy pongGameToView;
    public bool enableUpdate = true;
    public GameObject wallPref;
    public GameObject ballPref;
    public GameObject racketPref;
    public float wallThickness;
    public float racketThickness;
    public float ballRadius;

    private GameObject[] walls;
    private GameObject ball;
    private GameObject racketLeft;
    private GameObject racketRight;
    // Use this for initialization
    void Start () {
        Initialize();

    }
	
	// Update is called once per frame
	void Update () {
        if(enableUpdate)
            UpdateGraphics();

    }

    private void UpdateGraphics()
    {
        PongAcademy.GameState gameState = pongGameToView.CurrentGameState;

        walls[0].transform.localScale = new Vector3(pongGameToView.arenaSize.x, wallThickness, 1);
        walls[1].transform.localScale = new Vector3(pongGameToView.arenaSize.x, wallThickness, 1);
        walls[2].transform.localScale = new Vector3(wallThickness, pongGameToView.arenaSize.y, 1);
        walls[3].transform.localScale = new Vector3(wallThickness, pongGameToView.arenaSize.y, 1);

        walls[0].transform.position = new Vector3( 0, pongGameToView.arenaSize.y / 2 + wallThickness / 2, 0);
        walls[1].transform.position = new Vector3( 0, -pongGameToView.arenaSize.y / 2 - wallThickness / 2, 0);
        walls[2].transform.position = new Vector3(pongGameToView.arenaSize.x / 2 + wallThickness / 2, 0, 0);
        walls[3].transform.position = new Vector3(-pongGameToView.arenaSize.x / 2 - wallThickness / 2, 0, 0);

        ball.transform.localScale = Vector3.one * ballRadius * 2;
        ball.transform.position = new Vector3(gameState.ballPosition.x, gameState.ballPosition.y,0);

        racketLeft.transform.localScale = new Vector3(racketThickness, pongGameToView.racketWidth, 1);
        racketLeft.transform.position = new Vector3(pongGameToView.leftStartX - racketThickness / 2, gameState.leftY, 0);
        racketRight.transform.localScale = new Vector3(racketThickness, pongGameToView.racketWidth, 1);
        racketRight.transform.position = new Vector3(pongGameToView.rightStartX + racketThickness / 2, gameState.rightY, 0);
    }

    private void Initialize()
    {
        walls = new GameObject[4];
        walls[0] = Instantiate(wallPref);
        walls[1] = Instantiate(wallPref);
        walls[2] = Instantiate(wallPref);
        walls[3] = Instantiate(wallPref);

        ball = Instantiate(ballPref);
        racketLeft = Instantiate(racketPref);
        racketRight = Instantiate(racketPref);
    }
}
