using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using System.Linq;



// Class contains all necessary environment parameters
// to be defined and sent to external agent
#if ENABLE_TENSORFLOW
public enum BrainType
{
    Player,
    Heuristic,
    External,
    Internal
}
#else
public enum BrainType
{
Player,
Heuristic,
External,
}
#endif




public enum StateType
{
    discrete,
    continuous
}
;

/** Only need to be modified in the brain's inpector.
 * Defines what is the resolution of the camera
*/
[System.Serializable]
public struct resolution
{
    public int width;
    /**< \brief The width of the observation in pixels */
    public int height;
    /**< \brief The height of the observation in pixels */
    public bool blackAndWhite;
    /**< \brief If true, the image will be in black and white. 
     * If false, it will be in colors RGB */
}

/** Should be modified via the Editor Inspector.
 * Defines brain-specific parameters
*/
[System.Serializable]
public class BrainParameters
{
    public int stateSize = 1;
    /**< \brief If continuous : The length of the float vector that represents 
     * the state
     * <br> If discrete : The number of possible values the state can take*/
    public int actionSize = 1;
    /**< \brief If continuous : The length of the float vector that represents the action
     * <br> If discrete : The number of possible values the action can take*/
    public int memorySize = 0;
    /**< \brief The length of the float vector that holds the memory for the agent */
    public resolution[] cameraResolutions;
    /**<\brief  The list of observation resolutions for the brain */

    public string[] actionDescriptions;
    /**< \brief The list of strings describing what the actions correpond to */
    public StateType actionSpaceType = StateType.discrete;
    /**< \brief Defines if the action is discrete or continuous */
    public StateType stateSpaceType = StateType.continuous;
    /**< \brief Defines if the state is discrete or continuous */
}

/**
 * Contains all high-level Brain logic. 
 * Add this component to an empty GameObject in your scene and drag this 
 * GameObject into your Academy to make it a child in the hierarchy.
 * Contains a set of CoreBrains, which each correspond to a different method
 * for deciding actions.
 */
public class Brain : MonoBehaviour
{
    public BrainParameters brainParameters = new BrainParameters();
    /**< \brief Defines brain specific parameters such as the state size*/
    public BrainType brainType;
    /**<  \brief Defines what is the type of the brain : 
     * External / Internal / Player / Heuristic*/
    [HideInInspector]
    public Dictionary<int, Agent> agents = new Dictionary<int, Agent>();
    /**<  \brief Keeps track of the agents which subscribe to this brain*/

    [SerializeField]
    ScriptableObject[] CoreBrains;

    public CoreBrain coreBrain;
    /**<  \brief Reference to the current CoreBrain used by the brain*/

    //Ensures the coreBrains are not dupplicated with the brains
    [SerializeField]
    private int instanceID;

    /// Ensures the brain has an up to date array of coreBrains
    /** Is called when the inspector is modified and into InitializeBrain. 
	 * If the brain gameObject was just created, it generates a list of 
	 * coreBrains (one for each brainType) */
    public void UpdateCoreBrains()
    {

        // If CoreBrains is null, this means the Brain object was just 
        // instanciated and we create instances of each CoreBrain
        if (CoreBrains == null)
        {
            CoreBrains = new ScriptableObject[System.Enum.GetValues(typeof(BrainType)).Length];
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
            }

        }

        // If the length of CoreBrains does not match the number of BrainTypes, 
        // we increase the length of CoreBrains
        if (CoreBrains.Length < System.Enum.GetValues(typeof(BrainType)).Length)
        {
            ScriptableObject[] new_CoreBrains = new ScriptableObject[System.Enum.GetValues(typeof(BrainType)).Length];
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                if ((int)bt < CoreBrains.Length)
                {
                    new_CoreBrains[(int)bt] = CoreBrains[(int)bt];
                }
                else
                {
                    new_CoreBrains[(int)bt] = ScriptableObject.CreateInstance("CoreBrain" + bt.ToString());
                }
            }
            CoreBrains = new_CoreBrains;
        }

        // If the stored instanceID does not match the current instanceID, 
        // this means that the Brain GameObject was duplicated, and
        // we need to make a new copy of each CoreBrain
        if (instanceID != gameObject.GetInstanceID())
        {
            foreach (BrainType bt in System.Enum.GetValues(typeof(BrainType)))
            {
                CoreBrains[(int)bt] = ScriptableObject.Instantiate(CoreBrains[(int)bt]);
            }
            instanceID = gameObject.GetInstanceID();
        }

        // The coreBrain to display is the one defined in brainType
        coreBrain = (CoreBrain)CoreBrains[(int)brainType];

        coreBrain.SetBrain(this);
    }

    /// This is called by the Academy at the start of the environemnt.
    public void InitializeBrain()
    {
        UpdateCoreBrains();
        coreBrain.InitializeCoreBrain();

    }

    /// Collects the states of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> state}
    public Dictionary<int, List<float>> CollectStates()
    {
        Dictionary<int, List<float>> result = new Dictionary<int, List<float>>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            List<float> states = idAgent.Value.CollectState();
            if ((states.Count != brainParameters.stateSize) && (brainParameters.stateSpaceType == StateType.continuous ))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting {1} continuous states but received {2}.", idAgent.Value.gameObject.name, brainParameters.stateSize, states.Count));
            }
            if ((states.Count != 1) && (brainParameters.stateSpaceType == StateType.discrete ))
            {
                throw new UnityAgentsException(string.Format(@"The number of states does not match for agent {0}:
    Was expecting 1 discrete states but received {1}.", idAgent.Value.gameObject.name, states.Count));
            }
            result.Add(idAgent.Key, states);
        }
        return result;
    }

    /// Collects the observations of all the agents which subscribe to this 
    /// brain and returns a dictionary {id -> Camera}
    public Dictionary<int, List<Camera>> CollectObservations()
    {
        Dictionary<int, List<Camera>> result = new Dictionary<int, List<Camera>>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            List<Camera> observations = idAgent.Value.observations;
            if (observations.Count < brainParameters.cameraResolutions.Count())
            {
                throw new UnityAgentsException(string.Format(@"The number of observations does not match for agent {0}:
	Was expecting at least {1} observation but received {2}.", idAgent.Value.gameObject.name, brainParameters.cameraResolutions.Count(), observations.Count));
            }
            result.Add(idAgent.Key, observations);
        }
        return result;

    }

    /// Collects the rewards of all the agents which subscribe to this brain
    /// and returns a dictionary {id -> reward}
    public Dictionary<int, float> CollectRewards()
    {
        Dictionary<int, float> result = new Dictionary<int, float>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            result.Add(idAgent.Key, idAgent.Value.reward);
        }
        return result;
    }

    /// Collects the done flag of all the agents which subscribe to this brain
    ///  and returns a dictionary {id -> done}
    public Dictionary<int, bool> CollectDones()
    {
        Dictionary<int, bool> result = new Dictionary<int, bool>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            result.Add(idAgent.Key, idAgent.Value.done);
        }
        return result;
    }

    /// Collects the actions of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> action}
    public Dictionary<int, float[]> CollectActions()
    {
        Dictionary<int, float[]> result = new Dictionary<int, float[]>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            result.Add(idAgent.Key, idAgent.Value.agentStoredAction);
        }
        return result;
    }

    /// Collects the memories of all the agents which subscribe to this brain 
    /// and returns a dictionary {id -> memories}
    public Dictionary<int, float[]> CollectMemories()
    {
        Dictionary<int, float[]> result = new Dictionary<int, float[]>();
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            result.Add(idAgent.Key, idAgent.Value.memory);
        }
        return result;
    }

    /// Takes a dictionary {id -> memories} and sends the memories to the 
    /// corresponding agents
    public void SendMemories(Dictionary<int, float[]> memories)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            idAgent.Value.memory = memories[idAgent.Key];
        }
    }

    /// Takes a dictionary {id -> actions} and sends the actions to the 
    /// corresponding agents
    public void SendActions(Dictionary<int, float[]> actions)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            //Add a check here to see if the component was destroyed ?
            idAgent.Value.UpdateAction(actions[idAgent.Key]);
        }
    }

    /// Takes a dictionary {id -> values} and sends the values to the 
    /// corresponding agents
    public void SendValues(Dictionary<int, float> values)
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            //Add a check here to see if the component was destroyed ?
            idAgent.Value.value = values[idAgent.Key];
        }
    }

    ///Sets all the agents which subscribe to the brain to done
    public void SendDone()
    {
        foreach (KeyValuePair<int, Agent> idAgent in agents)
        {
            idAgent.Value.done = true;
        }
    }

    /// Uses coreBrain to call SendState on the CoreBrain
    public void SendState()
    {
        coreBrain.SendState();
    }

    /// Uses coreBrain to call decideAction on the CoreBrain
    public void DecideAction()
    {
        coreBrain.DecideAction();
    }

    /// \brief Is used by the Academy to send a step message to all the agents 
    /// which are not done
    public void Step()
    {
        List<Agent> agentsToIterate = agents.Values.ToList();
        foreach (Agent agent in agentsToIterate)
        {
            if (!agent.done)
            {
                agent.Step();
            }
        }
    }

    /// Is used by the Academy to reset the agents if they are done
    public void ResetIfDone()
    {
        List<Agent> agentsToIterate = agents.Values.ToList();
        foreach (Agent agent in agentsToIterate)
        {
            if (agent.done)
            {
                if (!agent.resetOnDone)
                {
                    agent.AgentOnDone();
                }
                else
                {
                    agent.Reset();
                }
            }
        }
    }

    /// Is used by the Academy to reset all agents 
    public void Reset()
    {
        List<Agent> agentsToIterate = agents.Values.ToList();
        foreach (Agent agent in agentsToIterate)
        {
            agent.Reset();
            agent.done = false;
        }
    }

    /// \brief Is used by the Academy reset the done flag and the rewards of the
    /// agents that subscribe to the brain
    public void ResetDoneAndReward()
    {
        foreach (Agent agent in agents.Values)
        {
            if (!agent.done || agent.resetOnDone)
            {
                agent.ResetReward();
                agent.done = false;
            }
        }
    }

    /** Contains logic for coverting a camera component into a Texture2D. */
    public Texture2D ObservationToTex(Camera camera, int width, int height)
    {
        Camera cam = camera;
        Rect oldRec = camera.rect;
        cam.rect = new Rect(0f, 0f, 1f, 1f);
        bool supportsAntialiasing = false;
        bool needsRescale = false;
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;
        var antiAliasing = (supportsAntialiasing) ? Mathf.Max(1, QualitySettings.antiAliasing) : 1;

        var finalRT =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var renderRT = (!needsRescale) ? finalRT :
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);

        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = renderRT;
        cam.targetTexture = renderRT;

        cam.Render();

        if (needsRescale)
        {
            RenderTexture.active = finalRT;
            Graphics.Blit(renderRT, finalRT);
            RenderTexture.ReleaseTemporary(renderRT);
        }

        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();
        cam.targetTexture = prevCameraRT;
        cam.rect = oldRec;
        RenderTexture.active = prevActiveRT;
        RenderTexture.ReleaseTemporary(finalRT);
        return tex;
    }

    /// Contains logic to convert the agent's cameras into observation list
    ///  (as list of float arrays)
    public List<float[,,,]> GetObservationMatrixList(List<int> agent_keys)
    {
        List<float[,,,]> observation_matrix_list = new List<float[,,,]>();
        Dictionary<int, List<Camera>> observations = CollectObservations();
        for (int obs_number = 0; obs_number < brainParameters.cameraResolutions.Length; obs_number++)
        {
            int width = brainParameters.cameraResolutions[obs_number].width;
            int height = brainParameters.cameraResolutions[obs_number].height;
            bool bw = brainParameters.cameraResolutions[obs_number].blackAndWhite;
            int pixels = 0;
            if (bw)
                pixels = 1;
            else
                pixels = 3;
            float[,,,] observation_matrix = new float[agent_keys.Count
            , height
            , width
            , pixels];
            int i = 0;
            foreach (int k in agent_keys)
            {
                Camera agent_obs = observations[k][obs_number];
                Texture2D tex = ObservationToTex(agent_obs, width, height);
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        Color c = tex.GetPixel(w, h);
                        if (!bw)
                        {
                            observation_matrix[i, tex.height - h - 1, w, 0] = c.r;
                            observation_matrix[i, tex.height - h - 1, w, 1] = c.g;
                            observation_matrix[i, tex.height - h - 1, w, 2] = c.b;
                        }
                        else
                        {
                            observation_matrix[i, tex.height - h - 1, w, 0] = (c.r + c.g + c.b) / 3;
                        }
                    }
                }
                UnityEngine.Object.DestroyImmediate(tex);
                Resources.UnloadUnusedAssets();
                i++;
            }
            observation_matrix_list.Add(observation_matrix);
        }
        return observation_matrix_list;
    }

}