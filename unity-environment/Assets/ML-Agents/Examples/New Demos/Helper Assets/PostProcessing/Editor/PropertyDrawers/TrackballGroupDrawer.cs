using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.PostProcessing;

namespace UnityEditor.PostProcessing
{
    [CustomPropertyDrawer(typeof(TrackballGroupAttribute))]
    sealed class TrackballGroupDrawer : PropertyDrawer
    {
        static Material s_Material;

        const int k_MinWheelSize = 80;
        const int k_MaxWheelSize = 256;

        bool m_ResetState;

        // Cached trackball computation methods (for speed reasons)
        static Dictionary<string, MethodInfo> m_TrackballMethods = new Dictionary<string, MethodInfo>();

        internal static int m_Size
        {
            get
            {
                int size = Mathf.FloorToInt(EditorGUIUtility.currentViewWidth / 3f) - 18;
                size = Mathf.Clamp(size, k_MinWheelSize, k_MaxWheelSize);
                return size;
            }
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (s_Material == null)
                s_Material = new Material(Shader.Find("Hidden/Post FX/UI/Trackball")) { hideFlags = HideFlags.HideAndDontSave };

            position = new Rect(position.x, position.y, position.width / 3f, position.height);
            int size = m_Size;
            position.x += 5f;

            var enumerator = property.GetEnumerator();
            while (enumerator.MoveNext())
            {
                var prop = enumerator.Current as SerializedProperty;
                if (prop == null || prop.propertyType != SerializedPropertyType.Color)
                    continue;

                OnWheelGUI(position, size, prop.Copy());
                position.x += position.width;
            }
        }

        void OnWheelGUI(Rect position, int size, SerializedProperty property)
        {
            if (Event.current.type == EventType.Layout)
                return;

            var value = property.colorValue;
            float offset = value.a;

            var wheelDrawArea = position;
            wheelDrawArea.height = size;

            if (wheelDrawArea.width > wheelDrawArea.height)
            {
                wheelDrawArea.x += (wheelDrawArea.width - wheelDrawArea.height) / 2.0f;
                wheelDrawArea.width = position.height;
            }

            wheelDrawArea.width = wheelDrawArea.height;

            float hsize = size / 2f;
            float radius = 0.38f * size;
            Vector3 hsv;
            Color.RGBToHSV(value, out hsv.x, out hsv.y, out hsv.z);

            if (Event.current.type == EventType.Repaint)
            {
                float scale = EditorGUIUtility.pixelsPerPoint;

                // Wheel texture
                var oldRT = RenderTexture.active;
                var rt = RenderTexture.GetTemporary((int)(size * scale), (int)(size * scale), 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                s_Material.SetFloat("_Offset", offset);
                s_Material.SetFloat("_DisabledState", GUI.enabled ? 1f : 0.5f);
                s_Material.SetVector("_Resolution", new Vector2(size * scale, size * scale / 2f));
                Graphics.Blit(null, rt, s_Material, EditorGUIUtility.isProSkin ? 0 : 1);
                RenderTexture.active = oldRT;

                GUI.DrawTexture(wheelDrawArea, rt);
                RenderTexture.ReleaseTemporary(rt);

                // Thumb
                var thumbPos = Vector2.zero;
                float theta = hsv.x * (Mathf.PI * 2f);
                float len = hsv.y * radius;
                thumbPos.x = Mathf.Cos(theta + (Mathf.PI / 2f));
                thumbPos.y = Mathf.Sin(theta - (Mathf.PI / 2f));
                thumbPos *= len;
                var thumbSize = FxStyles.wheelThumbSize;
                var thumbSizeH = thumbSize / 2f;
                FxStyles.wheelThumb.Draw(new Rect(wheelDrawArea.x + hsize + thumbPos.x - thumbSizeH.x, wheelDrawArea.y + hsize + thumbPos.y - thumbSizeH.y, thumbSize.x, thumbSize.y), false, false, false, false);
            }

            var bounds = wheelDrawArea;
            bounds.x += hsize - radius;
            bounds.y += hsize - radius;
            bounds.width = bounds.height = radius * 2f;
            hsv = GetInput(bounds, hsv, radius);
            value = Color.HSVToRGB(hsv.x, hsv.y, 1f);
            value.a = offset;

            // Luminosity booster
            position = wheelDrawArea;
            float oldX = position.x;
            float oldW = position.width;
            position.y += position.height + 4f;
            position.x += (position.width - (position.width * 0.75f)) / 2f;
            position.width = position.width * 0.75f;
            position.height = EditorGUIUtility.singleLineHeight;
            value.a = GUI.HorizontalSlider(position, value.a, -1f, 1f);

            // Advanced controls
            var data = Vector3.zero;

            if (TryGetDisplayValue(value, property, out data))
            {
                position.x = oldX;
                position.y += position.height;
                position.width = oldW / 3f;

                using (new EditorGUI.DisabledGroupScope(true))
                {
                    GUI.Label(position, data.x.ToString("F2"), EditorStyles.centeredGreyMiniLabel);
                    position.x += position.width;
                    GUI.Label(position, data.y.ToString("F2"), EditorStyles.centeredGreyMiniLabel);
                    position.x += position.width;
                    GUI.Label(position, data.z.ToString("F2"), EditorStyles.centeredGreyMiniLabel);
                    position.x += position.width;
                }
            }

            // Title
            position.x = oldX;
            position.y += position.height;
            position.width = oldW;
            GUI.Label(position, property.displayName, EditorStyles.centeredGreyMiniLabel);

            if (m_ResetState)
            {
                value = Color.clear;
                m_ResetState = false;
            }

            property.colorValue = value;
        }

        bool TryGetDisplayValue(Color color, SerializedProperty property, out Vector3 output)
        {
            output = Vector3.zero;
            MethodInfo method;

            if (!m_TrackballMethods.TryGetValue(property.name, out method))
            {
                var field = ReflectionUtils.GetFieldInfoFromPath(property.serializedObject.targetObject, property.propertyPath);

                if (!field.IsDefined(typeof(TrackballAttribute), false))
                    return false;

                var attr = (TrackballAttribute)field.GetCustomAttributes(typeof(TrackballAttribute), false)[0];
                const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
                method = typeof(ColorGradingComponent).GetMethod(attr.method, flags);
                m_TrackballMethods.Add(property.name, method);
            }

            if (method == null)
                return false;

            output = (Vector3)method.Invoke(property.serializedObject.targetObject, new object[] { color });
            return true;
        }

        static readonly int k_ThumbHash = "colorWheelThumb".GetHashCode();

        Vector3 GetInput(Rect bounds, Vector3 hsv, float radius)
        {
            var e = Event.current;
            var id = GUIUtility.GetControlID(k_ThumbHash, FocusType.Passive, bounds);

            var mousePos = e.mousePosition;
            var relativePos = mousePos - new Vector2(bounds.x, bounds.y);

            if (e.type == EventType.MouseDown && GUIUtility.hotControl == 0 && bounds.Contains(mousePos))
            {
                if (e.button == 0)
                {
                    var center = new Vector2(bounds.x + radius, bounds.y + radius);
                    float dist = Vector2.Distance(center, mousePos);

                    if (dist <= radius)
                    {
                        e.Use();
                        GetWheelHueSaturation(relativePos.x, relativePos.y, radius, out hsv.x, out hsv.y);
                        GUIUtility.hotControl = id;
                        GUI.changed = true;
                    }
                }
                else if (e.button == 1)
                {
                    e.Use();
                    GUI.changed = true;
                    m_ResetState = true;
                }
            }
            else if (e.type == EventType.MouseDrag && e.button == 0 && GUIUtility.hotControl == id)
            {
                e.Use();
                GUI.changed = true;
                GetWheelHueSaturation(relativePos.x, relativePos.y, radius, out hsv.x, out hsv.y);
            }
            else if (e.rawType == EventType.MouseUp && e.button == 0 && GUIUtility.hotControl == id)
            {
                e.Use();
                GUIUtility.hotControl = 0;
            }

            return hsv;
        }

        void GetWheelHueSaturation(float x, float y, float radius, out float hue, out float saturation)
        {
            float dx = (x - radius) / radius;
            float dy = (y - radius) / radius;
            float d = Mathf.Sqrt(dx * dx + dy * dy);
            hue = Mathf.Atan2(dx, -dy);
            hue = 1f - ((hue > 0) ? hue : (Mathf.PI * 2f) + hue) / (Mathf.PI * 2f);
            saturation = Mathf.Clamp01(d);
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return m_Size + 4f * 2f + EditorGUIUtility.singleLineHeight * 3f;
        }
    }
}
