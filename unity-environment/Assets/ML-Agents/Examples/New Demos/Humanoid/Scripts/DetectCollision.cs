using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DetectCollision : MonoBehaviour {

	public HumanoidAgent agentScript;
	void Start()
	{
		// agentScript = transform.root.GetComponenti<HumanoidAgent>();
	}
	void OnCollisionEnter(Collision col)
	{
		if(col.transform.CompareTag("ground") && !agentScript.fell)
		{
			agentScript.fell = true;

		}
	}

}
