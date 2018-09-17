using UnityEngine;
using System.IO;

namespace MLAgents
{
	[System.Serializable]
	public struct DemonstrationMetaData
	{
		public int numberExperiences;
		public int numberEpisodes;
	}
	
	[RequireComponent(typeof(Agent))]
	public class DemonstrationRecorder : MonoBehaviour
	{
		public bool record;
		public string demonstrationName;
		public int maxLength;
		private Agent recordingAgent;
		private string filePath;
		private DemonstrationMetaData metaData;

		private void Start ()
		{
			if (Application.isEditor && record)
			{
				recordingAgent = GetComponent<Agent>();
				if (!Directory.Exists("Assets/Demonstrations"))
				{
					Directory.CreateDirectory("Assets/Demonstrations");
				}

				metaData = new DemonstrationMetaData
				{
					numberEpisodes = 0,
					numberExperiences = 0
				};
				CreateDemonstrationFile();
			}
		}

		private void CreateDemonstrationFile()
		{
			var jsonParameters = JsonUtility.ToJson(recordingAgent.brain.brainParameters);

			var literalName = demonstrationName;
			filePath = "Assets/Demonstrations/" + literalName + ".demo";
			var uniqueNameCounter = 0;
			while (File.Exists(filePath))
			{
				literalName = demonstrationName + "_" + uniqueNameCounter;
				filePath = "Assets/Demonstrations/" + literalName + ".demo";
				uniqueNameCounter++;
			}
		
			var writer = File.CreateText(filePath);
			writer.Write(jsonParameters + '\n');
			writer.Close();
		}

		public void WriteExperience(AgentInfo info)
		{
			metaData.numberExperiences++;
			if (info.done)
			{
				metaData.numberEpisodes++;
			}
			var jsonInfo = JsonUtility.ToJson(info);
			var writer = new StreamWriter(filePath, true);
			writer.WriteLine(jsonInfo);
			writer.Close();
		}

		private void OnApplicationQuit()
		{
			if (Application.isEditor && record)
			{
				var jsonInfo = JsonUtility.ToJson(metaData);
				var writer = new StreamWriter(filePath, true);
				writer.WriteLine(jsonInfo);
				writer.Close();
			}
		}
	}
}
