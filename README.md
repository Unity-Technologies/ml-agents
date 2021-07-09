![Header](/doc_images/Dodgeball_TitleScreen.png)

# ML-Agents DodgeBall

## Overview

The [ML-Agents](https://github.com/Unity-Technologies/ml-agents) DodgeBall environment is a third-person cooperative shooter where players try to pick up as many balls as they can, then throw them at their opponents. It comprises two game modes: Elimination and Capture the Flag. In Elimination, each group tries to eliminate all members of the other group by hitting them with balls. In Capture the Flag, players try to steal the other team’s flag and bring it back to their base. In both modes, players can hold up to four balls, and dash to dodge incoming balls and go through hedges. You can find more information about the environment at the corresponding blog post[insert link].

This environment is intended to be used with the new features announced in [ML-Agents 2.0](https://blog.unity.com/technology/ml-agents-v20-release-now-supports-training-complex-cooperative-behaviors), namely cooperative behaviors and variable length observations. By using the [MA-POCA trainer](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Learning-Environment-Design-Agents.md#groups-for-cooperative-scenarios), [variable length observations](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Learning-Environment-Design-Agents.md#groups-for-cooperative-scenarios), and [self-play](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Learning-Environment-Design-Agents.md#teams-for-adversarial-scenarios), you can train teams of DodgeBall agents to play against each other. Trained agents are also provided in this project to play with, as both your allies and your opponents.

## Installation and Play

To open this repository, you will need to install the [Unity editor version 2020.2.6 or later](https://unity3d.com/get-unity/download).

Clone the `dodgeball-env` branch of this repository by running:
```
git clone --single-branch --branch dodgeball-env https://github.com/Unity-Technologies/ml-agents.git
```

Open the root folder in Unity. Then, navigate to `Assets/Dodgeball/Scenes/TitleScreen.unity`, open it, and hit the play button to play against pretrained agents. You can also build this scene (along with the `Elimination.unity` and `CaptureTheFlag.unity` scenes into a game build and play from there.
## Scenes

In `Assets/Dodgeball/Scenes/`, in addition to the title screen, six scenes are provided. They are:
* `X.unity`
* `X_Training.unity`
* `XVideoCapture.unity`

Where `X` is either Elimination or CaptureTheFlag, which corresponds to the two game modes.

The `X.unity` scenes are the playable scenes for human play against agent opponents. The `X_Training.unity` scenes are intended for training - more information on that below. Finally, the `XVideoCapture.unity` are scenes that let you watch a match between agents - they were used to record the videos below.
### Elimination

In the elimination scenes, four players face off against another team of four. Balls are dropped throughout the stage, and players must pick up balls and throw them at opponents. If a player is hit twice by an opponent, they are "out", and sent to the penalty podium in the top-center of the stage.

![EliminationVideo](/doc_images/ShorterElimination.gif)
### Capture the Flag

In the Capture the Flag scenes, the players must grab the enemy flag and return it to their base. But they can only score when their own flag is still safe at their base - denoted by a colored ring around the flag. If a player is hit by a ball in this mode, they are stunned in-place for ten seconds, and if they were holding the opponent's flag, they will drop it. Anyone can pick up the dropped flag - if a player picks up their own flag, it is returned to base.

![CTFVideo](/doc_images/ShorterCTF.gif)

## Training

ML-Agents DodgeBall was built using *ML-Agents Release 18* (Unity package 2.1.0-exp.1). We recommend the matching version of the Python trainers (Version 0.27.0) though newer trainers should work. See the [Releases Page](https://github.com/Unity-Technologies/ml-agents#releases--documentation) on the ML-Agents Github for more version information.

To train DodgeBall, in addition to downloading and opening this environment, you will need to [install the ML-Agents Python package](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Installation.md#install-the-mlagents-python-package). Follow the [getting started guide](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Getting-Started.md) for more information on how to use the ML-Agents trainers.

You will need to use either the `CaptureTheFlag_Training.unity` or `Elimination_Training.unity` scenes for training. Since training takes a *long* time, we recommend building these scenes into a Unity build.

A configuration YAML (`DodgeBall.yaml`) for ML-Agents is provided. You can uncomment and increase the number of environments (`num_envs`) depending on your computer's capabilities.

After tens of millions of steps (this will take many, many hours!) your agents will start to improve. As with any self-play run, you should observe your [ELO increase over time](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Using-Tensorboard.md#self-play).

[embed training videos from blog post here]

### Environment Parameters

To produce the results in the blog post, we used the default environment as it is in this repo. However, we also provide [environment parameters](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Training-ML-Agents.md#environment-parameters) to adjust reward functions and control the environment from the trainer. You may find it useful, for instance, to experiment with curriculums.

| **Setting**              | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ball_hold_bonus`| (default = `0.0`) A reward given to an agent at every timestep for each ball it is holding.|
| `is_capture_the_flag`| Set this parameter to 1 to override the scene's game mode setting, and change it to Capture the Flag. Set to 0 for Elimination.|
| `time_bonus_scale`| (default = `1.0` for Elimination, and `0.0` for CTF) Multiplier for negative reward given for taking too long to finish the game. Set to 1.0 for a -1.0 reward if it takes the maximum number of steps to finish the match.|
| `elimination_hit_reward`| (default = `0.1`) In Elimination, a reward given to an agent when it hits an opponent with a ball.|
| `stun_time` | (default = `10.0`) In Capture the Flag, the number of seconds an agent is stunned for when it is hit by a ball.|
| `opponent_has_flag_penalty`| (default = `0.0`) In Capture the Flag, a penalty (negative reward) given to the team at every timestep if an opponent has their flag. Use a negative value here. |
| `team_has_flag_bonus`| (default = `0.0`) In Capture the Flag, a reward given to the team at every timestep if one of the team members has the opponent's flag.|
| `return_flag_bonus`| (default = `0.0`) In Capture the Flag, a reward given to the team when it returns their own flag to their base, after it has been dropped by an opponent.|
| `ctf_hit_reward`| (default = `0.02`) In Capture the Flag, a reward given to an agent when it hits an opponent with a ball.|


## Extending DodgeBall

We'd love to hear how you're using DodgeBall - if you've customized the environment in an interesting way definitely let us know by emailing mlagents@unity3d.com.
## Reporting Issues

This environment is intended to provide an example of using ML-Agents in a cooperative/competitive game-like scenario, and it is not intended to be a finished game. While we will do our best to keep it updated, support of later versions of ML-Agents or additional content is not guaranteed.

If you do run into any issues or bugs with DodgeBall, please file an Issue on Github.

## License

[Apache 2.0 License](LICENSE.txt)


