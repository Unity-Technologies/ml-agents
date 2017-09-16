using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;


/**
 * Attach this component to a game object with an Agent component in order to
 * visulaize relevant information about your agent. 
*/
public class AgentMonitor : MonoBehaviour
{
    public bool fixedPosition;
    /**< \brief If true, the monitor will be fixed in the top left corner of 
     * the screen. If false, it will move with the position of the agent*/
    public float verticalOffset = 10f;
    /**< \brief If fixedPosition is false, defines how high above the agent 
     * the monitor is */
    [Header("Displayed Values")]
    public bool DisplayBrainName = true;
    /**< \brief If true, the name of the brain will be displayed in the monitor */
    public bool DisplayBrainType = true;
    /**< \brief If true, the type of the brain will be displayed in the monitor */
    public bool DisplayFrameCount = true;
    /**< \brief If true, the number of steps since the agent was reset will be displayed in the monitor */
    public bool DisplayCurrentReward = true;
    /**< \brief If true, the current reward of the agent will be displayed in the monitor */
    public bool DisplayMaxReward = true;
    /**< \brief If true, the maximum reward accros episodes will be displayed in the monitor */
    public bool DisplayState = false;
    /**< \brief If true, the current state of the agetn will be displayed in the monitor */
    public bool DisplayAction = false;
    /**< \brief If true, the current action of the agetn will be displayed in the monitor */

    private Agent agent;
    private Text texts;
    private GameObject monitor;
    private Slider slider;
    private Image fill;
    private float maxReward;


    void Start()
    {
        agent = gameObject.GetComponent<Agent>();

        GameObject canvas = GameObject.Find("AgentMonitorCanvas");
        if (canvas == null)
        {
            canvas = new GameObject();
            canvas.name = "AgentMonitorCanvas";
        }
        if (canvas.GetComponent<Canvas>() == null)
        {
            Canvas c = canvas.AddComponent<Canvas>();
            c.renderMode = RenderMode.ScreenSpaceOverlay;
            c.pixelPerfect = true;
        }
        if (canvas.GetComponent<CanvasScaler>() == null)
        {
            CanvasScaler cs = canvas.AddComponent<CanvasScaler>() as CanvasScaler;
            cs.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;

        }
        if (canvas.GetComponent<GraphicRaycaster>() == null)
        {
            canvas.AddComponent<GraphicRaycaster>();
        }

        monitor = Instantiate(Resources.Load("AgentMonitor") as GameObject);
        monitor.transform.SetParent(canvas.transform, false);

        slider = monitor.GetComponentInChildren<Slider>();
        fill = slider.GetComponentInChildren<Image>();

        texts = monitor.GetComponent<Text>();

        if (fixedPosition)
        {
            monitor.GetComponent<RectTransform>().anchorMin = new Vector2(0, 1);
            monitor.GetComponent<RectTransform>().anchorMax = new Vector2(0, 1);
            monitor.GetComponent<RectTransform>().pivot = new Vector2(0, 1);
            monitor.GetComponent<RectTransform>().position = new Vector3(0, monitor.GetComponent<RectTransform>().position.y, 0);
        }

    }

    void Update()
    {
        if (maxReward < agent.CummulativeReward)
        {
            maxReward = agent.CummulativeReward;
        }
        texts.text = "";
        if (DisplayBrainName)
        {
            texts.text += "Brain name : " + agent.brain.gameObject.name + "\n";
        }
        if (DisplayBrainType)
        {
            texts.text += "Mode : " + agent.brain.brainType + "\n";
        }
        if (DisplayFrameCount)
        {
            texts.text += "Frame Count :" + agent.stepCounter + "\n";
        }
        if (DisplayCurrentReward)
        {
            texts.text += "Reward : " + agent.CummulativeReward.ToString("F2") + "\n";
        }
        if (DisplayMaxReward)
        {
            texts.text += "Maximum Reward : " + maxReward.ToString("F2") + "\n";
        }
        if (DisplayState)
        {
            if (agent.brain.brainParameters.stateSpaceType == StateType.continuous)
            {
                texts.text += "State : ";
                foreach (float f in agent.CollectState())
                {
                    texts.text += f.ToString("F2") + " ";
                }
                texts.text += "\n";
            }
            else
            {
                texts.text += "State : " + ((int)agent.CollectState()[0]).ToString() + "\n";
            }
        }
        if (DisplayAction)
        {
            if (agent.brain.brainParameters.actionSpaceType == StateType.continuous)
            {
                texts.text += "Action : ";
                foreach (float f in agent.agentStoredAction)
                {
                    texts.text += f.ToString("F2") + " ";
                }
                texts.text += "\n";
            }
            else
            {
                texts.text += "State : " + ((int)agent.agentStoredAction[0]).ToString() + "\n";
            }
        }

        if (Mathf.Abs(agent.value) > slider.maxValue)
        {
            slider.maxValue = Mathf.Abs(agent.value);
        }
        slider.value = Mathf.Abs(agent.value);
        if (agent.value > 0)
        {
            fill.color = Color.green;
        }
        else
        {
            fill.color = Color.red;
        }

        if (!fixedPosition)
        {
            Vector3 cam2obj = gameObject.transform.position - Camera.main.transform.position;
            if (Vector3.Dot(cam2obj, Camera.main.transform.forward) < 0)
            {
                monitor.SetActive(false);
            }
            else
            {
                monitor.SetActive(true);
                monitor.transform.position = Camera.main.WorldToScreenPoint(agent.transform.position + new Vector3(0, verticalOffset, 0));
                monitor.transform.localScale = 20f * Mathf.Min(1, 1f / (Vector3.Dot(cam2obj, Camera.main.transform.forward))) * new Vector3(1, 1, 1);
            }
        }
    }

    void OnDestroy()
    {
        Destroy(monitor);
    }

}
