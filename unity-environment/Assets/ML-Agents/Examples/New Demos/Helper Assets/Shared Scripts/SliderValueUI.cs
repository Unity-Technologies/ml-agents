using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class SliderValueUI : MonoBehaviour {

	public Slider sliderToUse;
	public Text textUI;

	// Use this for initialization
	void Start () {
		textUI.text = sliderToUse.value.ToString();
	}
	
	// // Updat
	
	public void UpdateSliderValueUI(float val)
	{
		// textUI.text = val.ToString();
		textUI.text = System.String.Format("{0:0.00}", val);
	}
}
