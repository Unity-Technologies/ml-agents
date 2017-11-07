using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScoreScript : MonoBehaviour {

    TextMesh tm;
    int score = 0;

	// Use this for initialization
	void Start () {
        tm = gameObject.GetComponent<TextMesh>();
	}
	
	public void ResetScore()
    {
        score = 0;
        if (tm) tm.text = score.ToString();
    }

    public void IncrementScore()
    {
        score++;
        if (tm) tm.text = score.ToString();
    }
}
