using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Monitor is used to display information about the Agent within the Unity
    /// scene. Use the log function to add information to your monitor.
    /// </summary>
    public class Monitor : MonoBehaviour
    {
        /// <summary>
        /// The type of monitor the information must be displayed in.
        /// <slider> corresponds to a single rectangle whose width is given
        /// by a float between -1 and 1. (green is positive, red is negative)
        /// <hist> corresponds to n vertical sliders.
        /// <text> is a text field.
        /// <bar> is a rectangle of fixed length to represent the proportions
        /// of a list of floats.
        /// </summary>
        public enum DisplayType
        {
            INDEPENDENT,
            PROPORTION
        }

        /// <summary>
        /// Represents how high above the target the monitors will be.
        /// </summary>
        [HideInInspector] static public float verticalOffset = 3f;

        static bool isInstantiated;
        static GameObject canvas;
        static Dictionary<Transform, Dictionary<string, DisplayValue>> displayTransformValues;
        static Color[] barColors;

        struct DisplayValue
        {
            public float time;
            public string stringValue;
            public float floatValue;
            public float[] floatArrayValues;

            public enum ValueType
            {
                FLOAT,
                FLOATARRAY_INDEPENDENT,
                FLOATARRAY_PROPORTION,
                STRING
            }

            public ValueType valueType;
        }

        static GUIStyle keyStyle;
        static GUIStyle valueStyle;
        static GUIStyle greenStyle;
        static GUIStyle redStyle;
        static GUIStyle[] colorStyle;
        static bool initialized;

        /// <summary>
        /// Use the Monitor.Log static function to attach information to a transform.
        /// </summary>
        /// <returns>The log.</returns>
        /// <param name="key">The name of the information you wish to Log.</param>
        /// <param name="value">The string value you want to display.</param>
        /// <param name="target">The transform you want to attach the information to.
        /// </param>
        public static void Log(
            string key,
            string value,
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
                displayTransformValues[target] =
                    new Dictionary<string, DisplayValue>();
            }

            Dictionary<string, DisplayValue> displayValues =
                displayTransformValues[target];

            if (value == null)
            {
                RemoveValue(target, key);
                return;
            }

            if (!displayValues.ContainsKey(key))
            {
                var dv = new DisplayValue();
                dv.time = Time.timeSinceLevelLoad;
                dv.stringValue = value;
                dv.valueType = DisplayValue.ValueType.STRING;
                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    string max = (
                        displayValues
                            .Aggregate((l, r) => l.Value.time < r.Value.time ? l : r)
                            .Key
                    );
                    RemoveValue(target, max);
                }
            }
            else
            {
                DisplayValue dv = displayValues[key];
                dv.stringValue = value;
                dv.valueType = DisplayValue.ValueType.STRING;
                displayValues[key] = dv;
            }
        }

        /// <summary>
        /// Use the Monitor.Log static function to attach information to a transform.
        /// </summary>
        /// <returns>The log.</returns>
        /// <param name="key">The name of the information you wish to Log.</param>
        /// <param name="value">The float value you want to display.</param>
        /// <param name="target">The transform you want to attach the information to.
        /// </param>
        public static void Log(
            string key,
            float value,
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

            if (!displayValues.ContainsKey(key))
            {
                var dv = new DisplayValue();
                dv.time = Time.timeSinceLevelLoad;
                dv.floatValue = value;
                dv.valueType = DisplayValue.ValueType.FLOAT;
                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    string max = (
                        displayValues.Aggregate((l, r) => l.Value.time < r.Value.time ? l : r).Key);
                    RemoveValue(target, max);
                }
            }
            else
            {
                DisplayValue dv = displayValues[key];
                dv.floatValue = value;
                dv.valueType = DisplayValue.ValueType.FLOAT;
                displayValues[key] = dv;
            }
        }

        /// <summary>
        /// Use the Monitor.Log static function to attach information to a transform.
        /// </summary>
        /// <returns>The log.</returns>
        /// <param name="key">The name of the information you wish to Log.</param>
        /// <param name="value">The array of float you want to display.</param>
        /// <param name="displayType">The type of display.</param>
        /// <param name="target">The transform you want to attach the information to.
        /// </param>
        public static void Log(
            string key,
            float[] value,
            Transform target = null,
            DisplayType displayType = DisplayType.INDEPENDENT
        )
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

            if (!displayValues.ContainsKey(key))
            {
                var dv = new DisplayValue();
                dv.time = Time.timeSinceLevelLoad;
                dv.floatArrayValues = value;
                if (displayType == DisplayType.INDEPENDENT)
                {
                    dv.valueType = DisplayValue.ValueType.FLOATARRAY_INDEPENDENT;
                }
                else
                {
                    dv.valueType = DisplayValue.ValueType.FLOATARRAY_PROPORTION;
                }

                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    string max = (
                        displayValues.Aggregate((l, r) => l.Value.time < r.Value.time ? l : r).Key);
                    RemoveValue(target, max);
                }
            }
            else
            {
                DisplayValue dv = displayValues[key];
                dv.floatArrayValues = value;
                if (displayType == DisplayType.INDEPENDENT)
                {
                    dv.valueType = DisplayValue.ValueType.FLOATARRAY_INDEPENDENT;
                }
                else
                {
                    dv.valueType = DisplayValue.ValueType.FLOATARRAY_PROPORTION;
                }

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

            displayTransformValues = new Dictionary<Transform,
                Dictionary<string, DisplayValue>>();
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
                var origin = new Vector3(
                    Screen.width / 2 - keyPixelWidth, Screen.height);
                if (!(target == canvas.transform))
                {
                    Vector3 cam2obj = target.position - Camera.main.transform.position;
                    scale = Mathf.Min(
                        1,
                        20f / (Vector3.Dot(cam2obj, Camera.main.transform.forward)));
                    Vector3 worldPosition = Camera.main.WorldToScreenPoint(
                        target.position + new Vector3(0, verticalOffset, 0));
                    origin = new Vector3(
                        worldPosition.x - keyPixelWidth * scale, Screen.height - worldPosition.y);
                }

                keyPixelWidth *= scale;
                keyPixelHeight *= scale;
                paddingwidth *= scale;
                keyStyle.fontSize = (int) (keyPixelHeight * 0.8f);
                if (keyStyle.fontSize < 2)
                {
                    continue;
                }


                Dictionary<string, DisplayValue> displayValues = displayTransformValues[target];

                int index = 0;
                var orderedKeys = displayValues.Keys.OrderBy(x => -displayValues[x].time);
                float[] vals;
                GUIStyle s;
                foreach (string key in orderedKeys)
                {
                    keyStyle.alignment = TextAnchor.MiddleRight;
                    GUI.Label(
                        new Rect(
                            origin.x, origin.y - (index + 1) * keyPixelHeight,
                            keyPixelWidth, keyPixelHeight),
                        key,
                        keyStyle);
                    switch (displayValues[key].valueType)
                    {
                        case DisplayValue.ValueType.STRING:
                            valueStyle.alignment = TextAnchor.MiddleLeft;
                            GUI.Label(
                                new Rect(
                                    origin.x + paddingwidth + keyPixelWidth,
                                    origin.y - (index + 1) * keyPixelHeight,
                                    keyPixelWidth, keyPixelHeight),
                                displayValues[key].stringValue,
                                valueStyle);
                            break;
                        case DisplayValue.ValueType.FLOAT:
                            float sliderValue = displayValues[key].floatValue;
                            sliderValue = Mathf.Min(1f, sliderValue);
                            s = greenStyle;
                            if (sliderValue < 0)
                            {
                                sliderValue = Mathf.Min(1f, -sliderValue);
                                s = redStyle;
                            }

                            GUI.Box(
                                new Rect(
                                    origin.x + paddingwidth + keyPixelWidth,
                                    origin.y - (index + 0.9f) * keyPixelHeight,
                                    keyPixelWidth * sliderValue, keyPixelHeight * 0.8f),
                                GUIContent.none,
                                s);
                            break;

                        case DisplayValue.ValueType.FLOATARRAY_INDEPENDENT:
                            float histWidth = 0.15f;
                            vals = displayValues[key].floatArrayValues;
                            for (int i = 0; i < vals.Length; i++)
                            {
                                float value = Mathf.Min(vals[i], 1);
                                s = greenStyle;
                                if (value < 0)
                                {
                                    value = Mathf.Min(1f, -value);
                                    s = redStyle;
                                }

                                GUI.Box(
                                    new Rect(
                                        origin.x + paddingwidth + keyPixelWidth +
                                        (keyPixelWidth * histWidth + paddingwidth / 2) * i,
                                        origin.y - (index + 0.1f) * keyPixelHeight,
                                        keyPixelWidth * histWidth, -keyPixelHeight * value),
                                    GUIContent.none,
                                    s);
                            }

                            break;

                        case DisplayValue.ValueType.FLOATARRAY_PROPORTION:
                            float valsSum = 0f;
                            float valsCum = 0f;
                            vals = displayValues[key].floatArrayValues;
                            foreach (float f in vals)
                            {
                                valsSum += Mathf.Max(f, 0);
                            }

                            if (valsSum < float.Epsilon)
                            {
                                Debug.LogError(
                                    string.Format("The Monitor value for key {0} " +
                                                  "must be a list or array of " +
                                                  "positive values and cannot " +
                                                  "be empty.", key));
                            }
                            else
                            {
                                for (int i = 0; i < vals.Length; i++)
                                {
                                    float value = Mathf.Max(vals[i], 0) / valsSum;
                                    GUI.Box(
                                        new Rect(
                                            origin.x + paddingwidth +
                                            keyPixelWidth + keyPixelWidth * valsCum,
                                            origin.y - (index + 0.9f) * keyPixelHeight,
                                            keyPixelWidth * value, keyPixelHeight * 0.8f),
                                        GUIContent.none,
                                        colorStyle[i % colorStyle.Length]);
                                    valsCum += value;

                                }

                            }

                            break;
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
            barColors = new Color[6]
            {
                Color.magenta,
                Color.blue,
                Color.cyan,
                Color.green,
                Color.yellow,
                Color.red
            };
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
}
