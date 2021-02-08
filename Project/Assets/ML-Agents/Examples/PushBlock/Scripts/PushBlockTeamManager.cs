using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Extensions.Teams;
using Unity.MLAgents.Sensors;

public class PushBlockTeamManager : BaseTeamManager
{

    PushBlockEnvController m_envController;

    public PushBlockTeamManager(PushBlockEnvController envController)
    {
        m_envController = envController;
    }

    public override void OnTeamEpisodeBegin()
    {
        m_envController.ResetScene();
    }
}
