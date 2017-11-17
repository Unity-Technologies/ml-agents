using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MazeUI : MonoBehaviour {
    public MazeAgent agentRef;
    public MazeAcademy academyRef;
    public QLearningTrainer trainerRef;
    public Text stepsText;
    public Text episodeText;
    public Text winRateText;
    public Text episodeStepText;

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        episodeStepText.text = "Episode Steps: " + academyRef.currentStep.ToString();
        winRateText.text = "Win Rate: " + agentRef.winningRate100.Average.ToString();
        episodeText.text = "Episodes: " + academyRef.episodeCount.ToString();
        stepsText.text = "Steps: " + trainerRef.TotalSteps.ToString();
    }
}
