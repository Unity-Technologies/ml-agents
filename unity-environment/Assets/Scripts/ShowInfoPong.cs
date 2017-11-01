using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowInfoPong : MonoBehaviour {
    public Text stepsText;
    public Text episodeText;
    public Text leftWinText;
    public Text rightWinText;
    public Text leftWinPerText;
    public PongQNTrainer pongTrainer;
    public PongAgent leftAgent, rightAgent;

	// Use this for initialization
	void Start () {
		
	}

    // Update is called once per frame
    void Update() {
        stepsText.text = "Total Steps: " + pongTrainer.TotalSteps.ToString();
        episodeText.text = "Total Episodes: " + pongTrainer.Episodes.ToString();
        leftWinText.text = leftAgent.winCount.ToString();
        rightWinText.text = rightAgent.winCount.ToString();
        leftWinPerText.text = (leftAgent.winningRate100.Average).ToString();
    }
}
