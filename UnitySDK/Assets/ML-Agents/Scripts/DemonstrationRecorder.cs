using UnityEngine;
using System.IO;

namespace MLAgents
{
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
			if (Application.isEditor)
			{
				recordingAgent = GetComponent<Agent>();
				if (record)
				{
					if (!Directory.Exists("Assets/Demonstrations"))
					{
						Directory.CreateDirectory("Assets/Demonstrations");
					}
					CreateDemonstrationFile();
				}
			}
		}

		private void CreateDemonstrationFile()
		{
			var jsonParameters = JsonUtility.ToJson(recordingAgent.brain.brainParameters);

			var literalName = demonstrationName;
			filePath = "Assets/Demonstrations/" + literalName + ".demo";
			int counter = 0;
			while (File.Exists(filePath))
			{
				literalName = demonstrationName + "_" + counter;
				filePath = "Assets/Demonstrations/" + literalName + ".demo";
				counter++;
			}
		
			StreamWriter writer = File.CreateText(filePath);
			writer.Write(jsonParameters + '\n');
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
}
