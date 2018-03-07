//Read realtime reward data
//Attach this to any object (The academy is a good choice)
//In the editor assign paste the path to your python folder in this repo

using UnityEngine;
using System.Collections.Generic;
// using UnityEditor;
using System.IO;



public class ReadRewardData : MonoBehaviour
{
    
    [System.Serializable]
    public class RewardData
    {
        // int posIndex;
        public string brainName;
        public float currentMeanReward;
        public float currentStdDeviation;

    }

    [System.Serializable]
    public class BrainRewardData
    {
        public List<RewardData> rewardData;
    }

    public string pathToPythonFolder = "/Users/brandonh/unity_projects/ml-agents/python/";
    string pathToRewardFile;
    StreamReader reader;
    // public List<BrainRewardData> rewardData;
    public BrainRewardData rewardData = new BrainRewardData();
    // public BrainRewardData[] rewardData = new BrainRewardData[10];
    public Dictionary<string, RewardData> rewardDataDict = new Dictionary<string, RewardData>();


    void Start()
    {
        pathToRewardFile = pathToPythonFolder + "currentRewardData.json";
        InvokeRepeating("ReadData", 0, 5);
    }

    public void Load(string jsonString)
    {
        JsonUtility.FromJsonOverwrite(jsonString, rewardData);
        foreach (RewardData item in rewardData.rewardData)
        {
            rewardDataDict[item.brainName] = item;
            print(item.brainName);
        }
    }

    void ReadData()
    {
        reader = new StreamReader(pathToRewardFile);
        string json = reader.ReadToEnd();
        Load(json);
        reader.Close();
    }

}