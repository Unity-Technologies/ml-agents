using UnityEngine;

namespace MujocoUnity
{

    public class MujocoAgent : Agent
    {
        public bool FootHitTerrain;
        public bool NonFootHitTerrain;

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        FootHitTerrain = false;
        NonFootHitTerrain = false;
    }        
    public void OnTerrainCollision(GameObject other, GameObject terrain) {
        if (string.Compare(terrain.name, "Terrain", true) != 0)
            return;
        
        switch (other.name.ToLowerInvariant().Trim())
        {
            case "left_foot": // oai_humanoid
            case "right_foot": // oai_humanoid
            case "right_shin1": // oai_humanoid
            case "left_shin1": // oai_humanoid
            case "foot_geom": // oai_hopper  //oai_walker2d
            case "leg_geom": // oai_hopper //oai_walker2d
            case "leg_left_geom": // oai_walker2d
            case "foot_left_geom": //oai_walker2d
            case "foot_left_joint": //oai_walker2d
            case "foot_joint": //oai_walker2d
                FootHitTerrain = true;
                break;
            default:
                NonFootHitTerrain = true;
                break;
        }
    }         
    }
}