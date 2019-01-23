using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class TennisCanvas : MonoBehaviour
{

    public Text recentBlueWinrate;
    public Text recentLearnWinrate;
    public Text scoreGhost;
    public Text scoreLearn;

    public static string learnWinrate(){
        int recentLearnWins = TennisArea.last1000BrainResults.Where(result => result == 'L').Count();
        int recentGhostWins = TennisArea.last1000BrainResults.Where(result => result == 'G').Count();
        float winrate = recentLearnWins / (float)(recentLearnWins + recentGhostWins);
        return "Learn Winrate:" + winrate.ToString("F3");
    }

    public static string blueWinrate(){
        int recentAWins = TennisArea.last1000AgentResults.Where(result => result == 'A').Count();
        int recentBWins = TennisArea.last1000AgentResults.Where(result => result == 'B').Count();
        float winrate = recentAWins / (float)(recentAWins + recentBWins);
        return "Blue Winrate:" + winrate.ToString("F3");
    }

    public static double averagePasses(){
        if (TennisArea.last1000Passes.Count == 0)
            return -1;
        return TennisArea.last1000Passes.Average();
    }

    void Update()
    {
        scoreGhost.text = "Ghost:" + TennisArea.ghostScore;
        scoreLearn.text = "Learn:" + TennisArea.learningScore;

        recentLearnWinrate.text = learnWinrate();
        recentBlueWinrate.text = blueWinrate();
    }
}
