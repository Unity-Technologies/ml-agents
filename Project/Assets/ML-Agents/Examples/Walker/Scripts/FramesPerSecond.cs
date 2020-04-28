using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
public class FramesPerSecond : MonoBehaviour {

	// public TMP_Text fpsText;
	public int frameRange = 60;
	public int[] fpsBuffer;
	public int fpsBufferIndex;
	public int AverageFPS;

	// Use this for initialization
	void Start () {
		
	}
	
	void Update () {
		if (fpsBuffer == null || fpsBuffer.Length != frameRange) {
			InitializeBuffer();
		}
		UpdateBuffer();
		// CalculateFPS();
		// if(fpsText)
		// {
		// fpsText.text = AverageFPS.ToString();

		// }
	}


	void InitializeBuffer () {
		if (frameRange <= 0) {
			frameRange = 1;
		}
		fpsBuffer = new int[frameRange];
		fpsBufferIndex = 0;
	}

	void UpdateBuffer () {
		fpsBuffer[fpsBufferIndex++] = (int)(1f / Time.unscaledDeltaTime);
		if (fpsBufferIndex >= frameRange) {
			fpsBufferIndex = 0;
		}
	}

	public void CalculateFPS () 
	{
		int sum = 0;
		for (int i = 0; i < frameRange; i++) {
			sum += fpsBuffer[i];
		}
		AverageFPS = sum / frameRange;
	}
}
