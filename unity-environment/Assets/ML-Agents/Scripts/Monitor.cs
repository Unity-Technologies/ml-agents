using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Newtonsoft.Json;
using System.Linq;


/** The type of monitor the information must be displayed in.
 * <slider> corresponds to a slingle rectangle which width is given
 * by a float between -1 and 1. (green is positive, red is negative)
 * <hist> corresponds to n vertical sliders. 
 * <text> is a text field.
 * <bar> is a rectangle of fixed length to represent the proportions 
 * of a list of floats.
 */ 
public enum MonitorType
{
    slider,
    hist,
    text,
    bar
}

/** Monitor is used to display information. Use the log function to add
 * information to your monitor.
 */ 
public class Monitor : MonoBehaviour
{

    static bool isInstanciated;
    static GameObject canvas;

    private struct DisplayValue
    {
        public float time;
        public object value;
        public MonitorType monitorDisplayType;
    }

    static Dictionary<Transform, Dictionary<string,  DisplayValue>> displayTransformValues;
    static private Color[] barColors;
    [HideInInspector]
    static public float verticalOffset = 3f;
    /**< \brief This float represents how high above the target the monitors will be. */

    static GUIStyle keyStyle;
    static GUIStyle valueStyle;
    static GUIStyle greenStyle;
    static GUIStyle redStyle;
    static GUIStyle[] colorStyle;
    static bool initialized;


    /** Use the Monitor.Log static function to attach information to a transform.
     * If displayType is <text>, value can be any object. 
     * If sidplayType is <slider>, value must be a float.
     * If sidplayType is <hist>, value must be a List or Array of floats.
     * If sidplayType is <bar>, value must be a list or Array of positive floats.
     * Note that <slider> and <hist> caps values between -1 and 1.
     * @param key The name of the information you wish to Log.
     * @param value The value you want to display.
     * @param displayType The type of display.
     * @param target The transform you want to attach the information to.
     */ 
    public static void Log(
        string key, 
        object value, 
        MonitorType displayType = MonitorType.text, 
        Transform target = null)
    {



        if (!isInstanciated)
        {
            InstanciateCanvas();
            isInstanciated = true;

        }

        if (target == null)
        {
            target = canvas.transform;
        }

        if (!displayTransformValues.Keys.Contains(target))
        {
            displayTransformValues[target] = new Dictionary<string, DisplayValue>();
        }

        Dictionary<string, DisplayValue> displayValues = displayTransformValues[target];

        if (value == null)
        {
            RemoveValue(target, key);
            return;
        }
        if (!displayValues.ContainsKey(key))
        {
            DisplayValue dv = new DisplayValue();
            dv.time = Time.timeSinceLevelLoad;
            dv.value = value;
            dv.monitorDisplayType = displayType;
            displayValues[key] = dv;
            while (displayValues.Count > 20)
            {
                string max = displayValues.Aggregate((l, r) => l.Value.time < r.Value.time ? l : r).Key;
                RemoveValue(target, max);
            }
        }
        else
        {
            DisplayValue dv = displayValues[key];
            dv.value = value;
            displayValues[key] = dv;
        }
    }

    /** Remove a value from a monitor
     * @param target The transform to which the information is attached
     * @param key The key of the information you want to remove
     */ 
    public static void RemoveValue(Transform target, string key)
    {
        if (target == null)
        {
            target = canvas.transform;
        }
        if (displayTransformValues.Keys.Contains(target))
        {
            if (displayTransformValues[target].ContainsKey(key))
            {
                displayTransformValues[target].Remove(key);
                if (displayTransformValues[target].Keys.Count == 0)
                {
                    displayTransformValues.Remove(target);
                }
            }
        }

    }

    /** Remove all information from a monitor
     * @param target The transform to which the information is attached
     */ 
    public static void RemoveAllValues(Transform target)
    {
        if (target == null)
        {
            target = canvas.transform;
        }
        if (displayTransformValues.Keys.Contains(target))
        {
            displayTransformValues.Remove(target);
        }
    
    }

    /** Use SetActive to enable or disable the Monitor via script
     * @param active Set the Monitor's status to the value of active
     */ 
    public static void SetActive(bool active){
        if (!isInstanciated)
        {
            InstanciateCanvas();
            isInstanciated = true;

        }
        canvas.SetActive(active);

    }

    private static void InstanciateCanvas()
    {
        canvas = GameObject.Find("AgentMonitorCanvas");
        if (canvas == null)
        {
            canvas = new GameObject();
            canvas.name = "AgentMonitorCanvas";
            canvas.AddComponent<Monitor>();
        }
        displayTransformValues = new Dictionary<Transform, Dictionary< string , DisplayValue>>();

    }

    private float[] ToFloatArray(object input)
    {
        try
        {
            return JsonConvert.DeserializeObject<float[]>(
                JsonConvert.SerializeObject(input, Formatting.None));
        }
        catch
        {
        }
        try
        {
            return new float[1]
            {JsonConvert.DeserializeObject<float>(
                    JsonConvert.SerializeObject(input, Formatting.None))
            };
        }
        catch
        {
        }

        return new float[0];
    }

    void OnGUI()
    {
        if (!initialized)
        {
            Initialize();
            initialized = true;
        }

        var toIterate = displayTransformValues.Keys.ToList();
        foreach (Transform target in toIterate)
        {
            if (target == null)
            {
                displayTransformValues.Remove(target);
                continue;
            }
 
            float widthScaler = (Screen.width / 1000f);
            float keyPixelWidth = 100 * widthScaler;
            float keyPixelHeight = 20 * widthScaler;
            float paddingwidth = 10 * widthScaler;

            float scale = 1f;
            Vector2 origin = new Vector3(0, Screen.height);
            if (!(target == canvas.transform))
            {
                Vector3 cam2obj = target.position - Camera.main.transform.position;
                scale = Mathf.Min(1, 20f / (Vector3.Dot(cam2obj, Camera.main.transform.forward)));
                Vector3 worldPosition = Camera.main.WorldToScreenPoint(target.position + new Vector3(0, verticalOffset, 0));
                origin = new Vector3(worldPosition.x - keyPixelWidth * scale, Screen.height - worldPosition.y);
            }
            keyPixelWidth *= scale;
            keyPixelHeight *= scale;
            paddingwidth *= scale;
            keyStyle.fontSize = (int)(keyPixelHeight * 0.8f);
            if (keyStyle.fontSize < 2)
            {
                continue;
            }
                

            Dictionary<string, DisplayValue> displayValues = displayTransformValues[target];

            int index = 0;
            foreach (string key in displayValues.Keys.OrderBy(x => -displayValues[x].time))
            {
                keyStyle.alignment = TextAnchor.MiddleRight;
                GUI.Label(new Rect(origin.x, origin.y - (index + 1) * keyPixelHeight, keyPixelWidth, keyPixelHeight), key, keyStyle);
                if (displayValues[key].monitorDisplayType == MonitorType.text)
                {
                    valueStyle.alignment = TextAnchor.MiddleLeft;
                    GUI.Label(new Rect(
                            origin.x + paddingwidth + keyPixelWidth,
                            origin.y - (index + 1) * keyPixelHeight, 
                            keyPixelWidth, keyPixelHeight), 
                        JsonConvert.SerializeObject(displayValues[key].value, Formatting.None), valueStyle);

                }
                else if (displayValues[key].monitorDisplayType == MonitorType.slider)
                {
                    float sliderValue = 0f;
                    if (displayValues[key].value.GetType() == typeof(float))
                    {
                        sliderValue = (float)displayValues[key].value;
                    }
                    else
                    {
                        Debug.LogError(string.Format("The value for {0} could not be displayed as " +
                                "a slider because it is not a number.", key));
                    }

                    sliderValue = Mathf.Min(1f, sliderValue);
                    GUIStyle s = greenStyle;
                    if (sliderValue < 0)
                    {
                        sliderValue = Mathf.Min(1f, -sliderValue);
                        s = redStyle;
                    }
                    GUI.Box(new Rect(
                            origin.x + paddingwidth + keyPixelWidth,
                            origin.y - (index + 0.9f) * keyPixelHeight, 
                            keyPixelWidth * sliderValue, keyPixelHeight * 0.8f), 
                        GUIContent.none, s);

                }
                else if (displayValues[key].monitorDisplayType == MonitorType.hist)
                {
                    float histWidth = 0.15f;
                    float[] vals = ToFloatArray(displayValues[key].value);
                    for (int i = 0; i < vals.Length; i++)
                    {
                        float value = Mathf.Min(vals[i], 1);
                        GUIStyle s = greenStyle;
                        if (value < 0)
                        {
                            value = Mathf.Min(1f, -value);
                            s = redStyle;
                        }
                        GUI.Box(new Rect(
                                origin.x + paddingwidth + keyPixelWidth + (keyPixelWidth * histWidth + paddingwidth / 2) * i,
                                origin.y - (index + 0.1f) * keyPixelHeight, 
                                keyPixelWidth * histWidth, -keyPixelHeight * value), 
                            GUIContent.none, s);
                    }


                }
                else if (displayValues[key].monitorDisplayType == MonitorType.bar)
                {
                    float[] vals = ToFloatArray(displayValues[key].value);
                    float valsSum = 0f;
                    float valsCum = 0f; 
                    foreach (float f in vals)
                    {
                        valsSum += Mathf.Max(f, 0);
                    }
                    if (valsSum == 0)
                    {
                        Debug.LogError(string.Format("The Monitor value for key {0} must be "
                                + "a list or array of positive values and cannot be empty.", key));
                    }
                    else
                    {
                        for (int i = 0; i < vals.Length; i++)
                        {
                            float value = Mathf.Max(vals[i], 0) / valsSum;
                            GUI.Box(new Rect(
                                    origin.x + paddingwidth + keyPixelWidth + keyPixelWidth * valsCum,
                                    origin.y - (index + 0.9f) * keyPixelHeight, 
                                    keyPixelWidth * value, keyPixelHeight * 0.8f), 
                                GUIContent.none, colorStyle[i % colorStyle.Length]);
                            valsCum += value;

                        }

                    }

                }

                index++;
            }
        }
    }

    private void Initialize()
    {

        keyStyle = GUI.skin.label;
        valueStyle = GUI.skin.label;
        valueStyle.clipping = TextClipping.Overflow;
        valueStyle.wordWrap = false;



        barColors = new Color[6]{ Color.magenta, Color.blue, Color.cyan, Color.green, Color.yellow, Color.red };
        colorStyle = new GUIStyle[barColors.Length];
        for (int i = 0; i < barColors.Length; i++)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
            texture.SetPixel(0, 0, barColors[i]);
            texture.Apply();
            GUIStyle staticRectStyle = new GUIStyle();
            staticRectStyle.normal.background = texture;
            colorStyle[i] = staticRectStyle;
        }
        greenStyle = colorStyle[3];
        redStyle = colorStyle[5];
    }

}
