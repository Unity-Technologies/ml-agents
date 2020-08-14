using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.MLAgents
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
        /// </slider>
        /// <hist> corresponds to n vertical sliders. </hist>
        /// <text> is a text field. </text>
        /// <bar> is a rectangle of fixed length to represent the proportions </bar>
        /// of a list of floats.
        /// </summary>
        public enum DisplayType
        {
            Independent,
            Proportion
        }

        /// <summary>
        /// Represents how high above the target the monitors will be.
        /// </summary>
        public static float verticalOffset = 3f;

        static bool s_IsInstantiated;
        static GameObject s_Canvas;
        static Dictionary<Transform, Dictionary<string, DisplayValue>> s_DisplayTransformValues;

        /// <summary>
        /// Camera used to calculate GUI screen position relative to the target
        /// transform.
        /// </summary>
        static Dictionary<Transform, Camera> s_TransformCamera;

        static Color[] s_BarColors;

        struct DisplayValue
        {
            public float time;
            public string stringValue;
            public float floatValue;
            public float[] floatArrayValues;

            public enum ValueType
            {
                Float,
                FloatarrayIndependent,
                FloatarrayProportion,
                String
            }

            public ValueType valueType;
        }

        static GUIStyle s_KeyStyle;
        static GUIStyle s_ValueStyle;
        static GUIStyle s_GreenStyle;
        static GUIStyle s_RedStyle;
        static GUIStyle[] s_ColorStyle;
        static bool s_Initialized;

        /// <summary>
        /// Use the Monitor.Log static function to attach information to a transform.
        /// </summary>
        /// <returns>The log.</returns>
        /// <param name="key">The name of the information you wish to Log.</param>
        /// <param name="value">The string value you want to display.</param>
        /// <param name="target">The transform you want to attach the information to.
        /// </param>
        /// <param name="camera">Camera used to calculate GUI position relative to
        /// the target. If null, `Camera.main` will be used.</param>
        public static void Log(
            string key,
            string value,
            Transform target = null,
            Camera camera = null)
        {
            if (!s_IsInstantiated)
            {
                InstantiateCanvas();
                s_IsInstantiated = true;
            }
            if (s_Canvas == null)
            {
                return;
            }

            if (target == null)
            {
                target = s_Canvas.transform;
            }

            s_TransformCamera[target] = camera;

            if (!s_DisplayTransformValues.Keys.Contains(target))
            {
                s_DisplayTransformValues[target] =
                    new Dictionary<string, DisplayValue>();
            }

            var displayValues =
                s_DisplayTransformValues[target];

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
                dv.valueType = DisplayValue.ValueType.String;
                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    var max = (
                        displayValues
                            .Aggregate((l, r) => l.Value.time < r.Value.time ? l : r)
                            .Key
                    );
                    RemoveValue(target, max);
                }
            }
            else
            {
                var dv = displayValues[key];
                dv.stringValue = value;
                dv.valueType = DisplayValue.ValueType.String;
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
        /// <param name="camera">Camera used to calculate GUI position relative to
        /// the target. If null, `Camera.main` will be used.</param>
        public static void Log(
            string key,
            float value,
            Transform target = null,
            Camera camera = null)
        {
            if (!s_IsInstantiated)
            {
                InstantiateCanvas();
                s_IsInstantiated = true;
            }

            if (target == null)
            {
                target = s_Canvas.transform;
            }

            s_TransformCamera[target] = camera;

            if (!s_DisplayTransformValues.Keys.Contains(target))
            {
                s_DisplayTransformValues[target] = new Dictionary<string, DisplayValue>();
            }

            var displayValues = s_DisplayTransformValues[target];

            if (!displayValues.ContainsKey(key))
            {
                var dv = new DisplayValue();
                dv.time = Time.timeSinceLevelLoad;
                dv.floatValue = value;
                dv.valueType = DisplayValue.ValueType.Float;
                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    var max = (
                        displayValues.Aggregate((l, r) => l.Value.time < r.Value.time ? l : r).Key);
                    RemoveValue(target, max);
                }
            }
            else
            {
                var dv = displayValues[key];
                dv.floatValue = value;
                dv.valueType = DisplayValue.ValueType.Float;
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
        /// <param name="camera">Camera used to calculate GUI position relative to
        /// the target. If null, `Camera.main` will be used.</param>
        public static void Log(
            string key,
            float[] value,
            Transform target = null,
            DisplayType displayType = DisplayType.Independent,
            Camera camera = null
        )
        {
            if (!s_IsInstantiated)
            {
                InstantiateCanvas();
                s_IsInstantiated = true;
            }

            if (target == null)
            {
                target = s_Canvas.transform;
            }

            s_TransformCamera[target] = camera;

            if (!s_DisplayTransformValues.Keys.Contains(target))
            {
                s_DisplayTransformValues[target] = new Dictionary<string, DisplayValue>();
            }

            var displayValues = s_DisplayTransformValues[target];

            if (!displayValues.ContainsKey(key))
            {
                var dv = new DisplayValue();
                dv.time = Time.timeSinceLevelLoad;
                dv.floatArrayValues = value;
                if (displayType == DisplayType.Independent)
                {
                    dv.valueType = DisplayValue.ValueType.FloatarrayIndependent;
                }
                else
                {
                    dv.valueType = DisplayValue.ValueType.FloatarrayProportion;
                }

                displayValues[key] = dv;
                while (displayValues.Count > 20)
                {
                    var max = (
                        displayValues.Aggregate((l, r) => l.Value.time < r.Value.time ? l : r).Key);
                    RemoveValue(target, max);
                }
            }
            else
            {
                var dv = displayValues[key];
                dv.floatArrayValues = value;
                if (displayType == DisplayType.Independent)
                {
                    dv.valueType = DisplayValue.ValueType.FloatarrayIndependent;
                }
                else
                {
                    dv.valueType = DisplayValue.ValueType.FloatarrayProportion;
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
                target = s_Canvas.transform;
            }

            if (s_DisplayTransformValues.Keys.Contains(target))
            {
                if (s_DisplayTransformValues[target].ContainsKey(key))
                {
                    s_DisplayTransformValues[target].Remove(key);
                    if (s_DisplayTransformValues[target].Keys.Count == 0)
                    {
                        s_DisplayTransformValues.Remove(target);
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
                target = s_Canvas.transform;
            }

            if (s_DisplayTransformValues.Keys.Contains(target))
            {
                s_DisplayTransformValues.Remove(target);
            }
        }

        /// <summary>
        /// Use SetActive to enable or disable the Monitor via script
        /// </summary>
        /// <param name="active">Value to set the Monitor's status to.</param>
        public static void SetActive(bool active)
        {
            if (!s_IsInstantiated)
            {
                InstantiateCanvas();
                s_IsInstantiated = true;
            }

            if (s_Canvas != null)
            {
                s_Canvas.SetActive(active);
            }
        }

        /// Initializes the canvas.
        static void InstantiateCanvas()
        {
            s_Canvas = GameObject.Find("AgentMonitorCanvas");
            if (s_Canvas == null)
            {
                s_Canvas = new GameObject();
                s_Canvas.name = "AgentMonitorCanvas";
                s_Canvas.AddComponent<Monitor>();
            }

            s_DisplayTransformValues = new Dictionary<Transform,
                                                      Dictionary<string, DisplayValue>>();

            s_TransformCamera = new Dictionary<Transform, Camera>();
        }

        void OnGUI()
        {
            if (!s_Initialized)
            {
                Initialize();
                s_Initialized = true;
            }

            var toIterate = s_DisplayTransformValues.Keys.ToList();
            foreach (var target in toIterate)
            {
                if (target == null)
                {
                    s_DisplayTransformValues.Remove(target);
                    continue;
                }

                // get camera
                var cam = s_TransformCamera[target];
                if (cam == null)
                {
                    cam = Camera.main;
                }

                var widthScaler = (Screen.width / 1000f);
                var keyPixelWidth = 100 * widthScaler;
                var keyPixelHeight = 20 * widthScaler;
                var paddingWidth = 10 * widthScaler;

                var scale = 1f;
                var origin = new Vector3(
                    Screen.width / 2.0f - keyPixelWidth, Screen.height);
                if (!(target == s_Canvas.transform))
                {
                    var camTransform = cam.transform;
                    var position = target.position;
                    var cam2Obj = position - camTransform.position;
                    scale = Mathf.Min(
                        1,
                        20f / (Vector3.Dot(cam2Obj, camTransform.forward)));
                    var worldPosition = cam.WorldToScreenPoint(
                        position + new Vector3(0, verticalOffset, 0));
                    origin = new Vector3(
                        worldPosition.x - keyPixelWidth * scale, Screen.height - worldPosition.y);
                }

                keyPixelWidth *= scale;
                keyPixelHeight *= scale;
                paddingWidth *= scale;
                s_KeyStyle.fontSize = (int)(keyPixelHeight * 0.8f);
                if (s_KeyStyle.fontSize < 2)
                {
                    continue;
                }


                var displayValues = s_DisplayTransformValues[target];

                var index = 0;
                var orderedKeys = displayValues.Keys.OrderBy(x => -displayValues[x].time);
                foreach (var key in orderedKeys)
                {
                    s_KeyStyle.alignment = TextAnchor.MiddleRight;
                    GUI.Label(
                        new Rect(
                            origin.x, origin.y - (index + 1) * keyPixelHeight,
                            keyPixelWidth, keyPixelHeight),
                        key,
                        s_KeyStyle);
                    float[] vals;
                    GUIStyle s;
                    switch (displayValues[key].valueType)
                    {
                        case DisplayValue.ValueType.String:
                            s_ValueStyle.alignment = TextAnchor.MiddleLeft;
                            GUI.Label(
                                new Rect(
                                    origin.x + paddingWidth + keyPixelWidth,
                                    origin.y - (index + 1) * keyPixelHeight,
                                    keyPixelWidth, keyPixelHeight),
                                displayValues[key].stringValue,
                                s_ValueStyle);
                            break;
                        case DisplayValue.ValueType.Float:
                            var sliderValue = displayValues[key].floatValue;
                            sliderValue = Mathf.Min(1f, sliderValue);
                            s = s_GreenStyle;
                            if (sliderValue < 0)
                            {
                                sliderValue = Mathf.Min(1f, -sliderValue);
                                s = s_RedStyle;
                            }

                            GUI.Box(
                                new Rect(
                                    origin.x + paddingWidth + keyPixelWidth,
                                    origin.y - (index + 0.9f) * keyPixelHeight,
                                    keyPixelWidth * sliderValue, keyPixelHeight * 0.8f),
                                GUIContent.none,
                                s);
                            break;

                        case DisplayValue.ValueType.FloatarrayIndependent:
                            const float histWidth = 0.15f;
                            vals = displayValues[key].floatArrayValues;
                            for (var i = 0; i < vals.Length; i++)
                            {
                                var value = Mathf.Min(vals[i], 1);
                                s = s_GreenStyle;
                                if (value < 0)
                                {
                                    value = Mathf.Min(1f, -value);
                                    s = s_RedStyle;
                                }

                                GUI.Box(
                                    new Rect(
                                        origin.x + paddingWidth + keyPixelWidth +
                                        (keyPixelWidth * histWidth + paddingWidth / 2) * i,
                                        origin.y - (index + 0.1f) * keyPixelHeight,
                                        keyPixelWidth * histWidth, -keyPixelHeight * value),
                                    GUIContent.none,
                                    s);
                            }

                            break;

                        case DisplayValue.ValueType.FloatarrayProportion:
                            var valsSum = 0f;
                            var valsCum = 0f;
                            vals = displayValues[key].floatArrayValues;
                            foreach (var f in vals)
                            {
                                valsSum += Mathf.Max(f, 0);
                            }

                            if (valsSum < float.Epsilon)
                            {
                                Debug.LogError(
                                    $"The Monitor value for key {key} " +
                                    "must be a list or array of " +
                                    "positive values and cannot " +
                                    "be empty.");
                            }
                            else
                            {
                                for (var i = 0; i < vals.Length; i++)
                                {
                                    var value = Mathf.Max(vals[i], 0) / valsSum;
                                    GUI.Box(
                                        new Rect(
                                            origin.x + paddingWidth +
                                            keyPixelWidth + keyPixelWidth * valsCum,
                                            origin.y - (index + 0.9f) * keyPixelHeight,
                                            keyPixelWidth * value, keyPixelHeight * 0.8f),
                                        GUIContent.none,
                                        s_ColorStyle[i % s_ColorStyle.Length]);
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
            s_KeyStyle = GUI.skin.label;
            s_ValueStyle = GUI.skin.label;
            s_ValueStyle.clipping = TextClipping.Overflow;
            s_ValueStyle.wordWrap = false;
            s_BarColors = new[]
            {
                Color.magenta,
                Color.blue,
                Color.cyan,
                Color.green,
                Color.yellow,
                Color.red
            };
            s_ColorStyle = new GUIStyle[s_BarColors.Length];
            for (var i = 0; i < s_BarColors.Length; i++)
            {
                var texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
                texture.SetPixel(0, 0, s_BarColors[i]);
                texture.Apply();
                var staticRectStyle = new GUIStyle();
                staticRectStyle.normal.background = texture;
                s_ColorStyle[i] = staticRectStyle;
            }

            s_GreenStyle = s_ColorStyle[3];
            s_RedStyle = s_ColorStyle[5];
        }
    }
}
