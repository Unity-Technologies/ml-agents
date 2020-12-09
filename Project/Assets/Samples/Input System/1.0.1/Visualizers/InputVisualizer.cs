using System;

// Some fields assigned through only through serialization.
#pragma warning disable CS0649

namespace UnityEngine.InputSystem.Samples
{
    /// <summary>
    /// Base class for <see cref="InputActionVisualizer"/> and <see cref="InputControlVisualizer"/>.
    /// Not meant to be extended outside of input system.
    /// </summary>
    public abstract class InputVisualizer : MonoBehaviour
    {
        protected void OnEnable()
        {
            ResolveParent();
        }

        protected void OnDisable()
        {
            m_Parent = null;
            m_Visualizer = null;
        }

        protected void OnGUI()
        {
            if (Event.current.type != EventType.Repaint)
                return;

            // If we have a parent, offset our rect by the parent.
            var rect = m_Rect;
            if (m_Parent != null)
                rect.position += m_Parent.m_Rect.position;

            if (m_Visualizer != null)
                m_Visualizer.OnDraw(rect);
            else
                VisualizationHelpers.DrawRectangle(rect, new Color(1, 1, 1, 0.1f));

            // Draw label, if we have one.
            if (!string.IsNullOrEmpty(m_Label))
            {
                if (m_LabelContent == null)
                    m_LabelContent = new GUIContent(m_Label);
                if (s_LabelStyle == null)
                {
                    s_LabelStyle = new GUIStyle();
                    s_LabelStyle.normal.textColor = Color.yellow;
                }

                ////FIXME: why does CalcSize not calculate the rect width correctly?
                var labelSize = s_LabelStyle.CalcSize(m_LabelContent);
                var labelRect = new Rect(rect.x + 4, rect.y, labelSize.x + 4, rect.height);

                s_LabelStyle.Draw(labelRect, m_LabelContent, false, false, false, false);
            }
        }

        protected void OnValidate()
        {
            ResolveParent();
            m_LabelContent = null;
        }

        protected void ResolveParent()
        {
            var parentTransform = transform.parent;
            if (parentTransform != null)
                m_Parent = parentTransform.GetComponent<InputControlVisualizer>();
        }

        [SerializeField] internal string m_Label;
        [SerializeField] internal int m_HistorySamples = 500;
        [SerializeField] internal float m_TimeWindow = 3;
        [SerializeField] internal Rect m_Rect = new Rect(10, 10, 300, 30);

        [NonSerialized] internal GUIContent m_LabelContent;
        [NonSerialized] internal VisualizationHelpers.Visualizer m_Visualizer;
        [NonSerialized] internal InputVisualizer m_Parent;

        private static GUIStyle s_LabelStyle;
    }
}
