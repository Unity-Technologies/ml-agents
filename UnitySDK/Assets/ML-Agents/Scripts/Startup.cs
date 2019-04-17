using System;
using System.IO;
using Grpc.Core;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace MLAgents
{
    public class Startup: MonoBehaviour
    {
        void Awake()
        {   
            string sceneName = Environment.GetEnvironmentVariable("SCENE_NAME");


            if (SceneUtility.GetBuildIndexByScenePath(sceneName) >= 0)
            {
                SceneManager.LoadSceneAsync(sceneName);
            }
            else
            {
                throw new ArgumentException("The scene " + sceneName.ToString() + " doesn't exist within your build. ");
            }
            return;
        }

        void Update()
        {
            
        }
    }
}