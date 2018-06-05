using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if ENABLE_TENSORFLOW
using TensorFlow;
#endif

public class TensorflowTester : MonoBehaviour {
#if ENABLE_TENSORFLOW
	TFGraph graph;
	TFSession session;
	public TextAsset graphModelGrid;
	
	// Use this for initialization
	void Start () {
		
		graph = new TFGraph();
		graph.Import(graphModelGrid.bytes);
		session = new TFSession(graph);
		
	}
	
	// Update is called once per frame
	void Update () {
		var runner = session.GetRunner();
		float[,,,] visual = new float[1,84,84,3];
		runner.Fetch(graph["action"][0]);
		runner.AddInput(graph["visual_observation_0"][0], visual );
//		runner.AddInput(graph["batch_size"][0], new int[] { 0 });
		TFTensor[] networkOutput;
		networkOutput = runner.Run();
		long[,] output = networkOutput[0].GetValue() as long[,];
		Debug.Log(output[0,0]);
	}
	
#endif
}
