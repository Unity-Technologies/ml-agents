using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Docker.DotNet;
using System.Buffers;
using System;
using Docker.DotNet.Models;
using System.IO;

public class DockerTest : MonoBehaviour {

	// Use this for initialization
	void Start ()
	{
		Test();
	}

	private async void Test()
	{
		DockerClient client = new DockerClientConfiguration(
//				new Uri("http://ubuntu-docker.cloudapp.net:4243"))
			new Uri("localhost:5004"))
			.CreateClient();
//		client.Containers.StartContainerAsync("39e3317fd25", new ContainerStartParameters(){});
		using (FileStream stream = File.Open(Path.GetFullPath("..") + "/Dockerfile", FileMode.Open))
		{
			await client.Images.BuildImageFromDockerfileAsync(stream, new ImageBuildParameters
			{
				Tags = new List<string>{"TestFromUnity"}
			});
		}
		await client.Containers.StartContainerAsync("TestFromUnity", new ContainerStartParameters()
		{
			
		});
		Debug.Log("Finished");
		
		IList<ContainerListResponse> containers = await client.Containers.ListContainersAsync(
			new ContainersListParameters(){
				Limit = 10,
			});
		foreach (ContainerListResponse clr in containers)
		{
			Debug.Log(clr.ID);
		}
		
	}
	
	// Update is called once per frame
	void Update () {
//		var client = new DockerClientConfiguration()
		
	}
}
