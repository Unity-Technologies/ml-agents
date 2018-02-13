# Using Docker For ML Agents (Experimental)

We are currently offering an experimental solution for people who'd like to do training or inference using docker. This setup currently forces both python and Unity to rely on _only_ the CPU for computation purposes. So we don't support environments such as [GridWorld](Example-Environments.md#gridworld) which use visual observations for training.

## Setup

- Install [docker](https://www.docker.com/community-edition#/download) if you don't have it setup on your machine. 

- Since Docker runs a container in an environment that is isolated from the host machine, we will be using a mounted directory, e.g. `unity-volume` in your host machine in order to share data, e.g. the Unity executable, curriculum files and tensorflow graph.


## Usage

- Docker typically runs a container sharing a (linux) kernel with the host machine, this means that the 
Unity environment **has** to be built for the **linux platform**. Please select the architecture to be `x86_64` and choose the build to be `headless` (_this is important because we are running it in a container that does not have graphics drivers installed_). 
Save the generated environment in the directory to be mounted (e.g. we have conveniently created an empty directory called at the top level `unity-volume`). Ensure that  
`unity-volume/<environment-name>.x86_64` and `unity-volume/environment-name_Data`. So for example, `<environment_name>` might be `3Dball` and you might want to ensure that `unity-volume/3Dball.x86_64` and `unity-volume/3Dball_Data` are both present in the directory `unity-volume`.


- Make sure the docker engine is running on your machine, then build the docker container by running `docker build  -t <image_name> .` . in the top level of the source directory. Replace `<image_name>` by the name of the image that you want to use, e.g. `balance.ball.v0.1`.

- Run the container:
```

docker run --mount type=bind,source="$(pwd)"/unity-volume,target=/unity-volume \
	 <tag-name>:latest <environment-name> \
	 --docker-target-name=unity-volume 
	 --train --run-id=<run-id>
```

For our balance ball, example this would be:

- Run the container:
```

docker run --mount type=bind,source="$(pwd)"/unity-volume,target=/unity-volume \
	 balance.ball.v0.1:latest 3Dball \
	 --docker-target-name=unity-volume 
	 --train --run-id=<run-id>
```

**Note** The docker target volume name, `unity-volume` must be passed to ML-Agents as an argument using the `--docker-target-name` option. The output will be stored in mounted directory. 


