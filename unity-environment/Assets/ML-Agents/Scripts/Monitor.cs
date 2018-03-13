using System.Collections.Generic;
using System.Linq;

using UnityEngine;

using Newtonsoft.Json;

/// <summary>
/// The type of monitor the information must be displayed in.
/// <slider> corresponds to a single rectangle whose width is given
///     by a float between -1 and 1. (green is positive, red is negative)
/// <hist> corresponds to n vertical sliders.
/// <text> is a text field.
/// <bar> is a rectangle of fixed length to represent the proportions
///     of a list of floats.
/// </summary>
public enum MonitorType
{
    slider,
    hist,
    text,
    bar
}

/// <summary>
/// Monitor is used to display information about the Agent within the Unity
/// scene. Use the log function to add information to your monitor.
/// </summary>
public class Monitor : MonoBehaviour
{
    /// <summary>
    /// Represents how high above the target the monitors will be.
    /// </summary>
    [HideInInspector]
    static public float verticalOffset = 3f;

    static bool isInstantiated;
    static GameObject canvas;
    static Dictionary<Transform, Dictionary<string, DisplayValue>> displayTransformValues;
    static Color[] barColors;

    struct DisplayValue
    {
        public float time;
        public object value;
        public MonitorType monitorDisplayType;
    }

    static GUIStyle keyStyle;
    static GUIStyle valueStyle;
    static GUIStyle greenStyle;
    static GUIStyle redStyle;
    static GUIStyle[] colorStyle;
    static bool initialized;

    /// <summary>
    /// Use the Monitor.Log static function to attach information to a transform.
    /// If displayType is <text>, value can be any object. 
    /// If sidplayType is <slider>, value must be a float.
    /// If sidplayType is <hist>, value must be a List or Array of floats.
    /// If sidplayType is <bar>, value must be a list or Array of positive floats.
    /// Note that <slider> and <hist> caps values between -1 and 1.
    /// </summary>
    /// <returns>The log.</returns>
    /// <param name="key">The name of the information you wish to Log.</param>
    /// <param name="value">The value you want to display.</param>
    /// <param name="displayType">The type of display.</param>
    /// <param name="target">
    /// The transform you want to attach the information to.
    /// </param>
    public static void Log(
        string key,
        object value,
        MonitorType displayType = MonitorType.text,
        Transform target = null)
    {
        if (!isInstantiated)
        {
            InstantiateCanvas();
            isInstantiated = true;

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
            var dv = new DisplayValue();
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

    /// <summary>
    /// Remove a value from a monitor.
    /// </summary>
    /// <param name="target">
    /// The transform to which the information is attached.
    /// </param>
    /// <param name="key">The key of the information you want to remove.</param>
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

    /// <summary>
    /// Remove all information from a monitor.
    /// </summary>
    /// <param name="target">
    /// The transform to which the information is attached.
    /// </param>
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

    /// <summary>
    /// Use SetActive to enable or disable the Monitor via script
    /// </summary>
    /// <param name="active">Value to set the Monitor's status to.</param>
    public static void SetActive(bool active)
    {
        if (!isInstantiated)
        {
            InstantiateCanvas();
            isInstantiated = true;

        }
        if (canvas != null)
        {
            canvas.SetActive(active);
        }

    }

    /// Initializes the canvas.
    static void InstantiateCanvas()
    {
        canvas = GameObject.Find("AgentMonitorCanvas");
        if (canvas == null)
        {
            canvas = new GameObject();
            canvas.name = "AgentMonitorCanvas";
            canvas.AddComponent<Monitor>();
        }
        displayTransformValues = new Dictionary<Transform, Dictionary<string, DisplayValue>>();
    }

    /// Convert the input object to a float array. Returns a float[0] if the
    /// conversion process fails.
    float[] ToFloatArray(object input)
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

    /// <summary> <inheritdoc/> </summary>
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
            var origin = new Vector3(Screen.width / 2 - keyPixelWidth, Screen.height);
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
                    if (displayValues[key].value is float)
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

    /// Helper method used to initialize the GUI. Called once.
    void Initialize()
    {
        keyStyle = GUI.skin.label;
        valueStyle = GUI.skin.label;
        valueStyle.clipping = TextClipping.Overflow;
        valueStyle.wordWrap = false;
        barColors = new Color[6] { Color.magenta, Color.blue, Color.cyan, Color.green, Color.yellow, Color.red };
        colorStyle = new GUIStyle[barColors.Length];
        for (int i = 0; i < barColors.Length; i++)
        {
            var texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
            texture.SetPixel(0, 0, barColors[i]);
            texture.Apply();
            var staticRectStyle = new GUIStyle();
            staticRectStyle.normal.background = texture;
            colorStyle[i] = staticRectStyle;
        }
        greenStyle = colorStyle[3];
        redStyle = colorStyle[5];
    }
}
