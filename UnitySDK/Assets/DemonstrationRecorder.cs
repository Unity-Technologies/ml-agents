using UnityEngine;
using UnityEditor;
using System.IO;
using MLAgents;

[RequireComponent(typeof(Agent))]
public class DemonstrationRecorder : MonoBehaviour
{
	public bool record;
	public string demonstrationName;
	public int maxLength;
	private Agent recordingAgent;
	private string filePath;

	private void Start ()
	{
		recordingAgent = GetComponent<Agent>();
		if (!Directory.Exists("Assets/Demonstrations"))
		{
			Directory.CreateDirectory("Assets/Demonstrations");
		}
		CreateDemonstrationFile();
	}

	private void CreateDemonstrationFile()
	{
		var jsonParameters = JsonUtility.ToJson(recordingAgent.brain.brainParameters);

		var literalName = demonstrationName;
		filePath = "Assets/Demonstrations/" + literalName + ".json";
		int counter = 0;
		while (File.Exists(filePath))
		{
			literalName = demonstrationName + "_" + counter;
			filePath = "Assets/Demonstrations/" + literalName + ".json";
			counter++;
		}
		
		StreamWriter writer = File.CreateText(filePath);
		writer.Write(jsonParameters);
		writer.Close();
	}

	public void WriteExperience(AgentInfo info)
	{
		var jsonInfo = JsonUtility.ToJson(info);
		StreamWriter writer = new StreamWriter(filePath, true);
		writer.WriteLine(jsonInfo);
		writer.Close();
	}
}
