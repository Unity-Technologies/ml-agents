using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.MultiAgent;
using Unity.MLAgents.Sensors;

using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;
using UnityEngine.Serialization;

public class PushBlockAgentGroup : BaseMultiAgentGroup
{

    PushBlockEnvController m_envController;
    Action ResetScene;

    public PushBlockAgentGroup(PushBlockEnvController envController)
    {
        ResetScene = envController.ResetScene;
    }

    public override void OnGroupEpisodeBegin()
    {
        ResetScene.Invoke();
    }
}
