using System;
using System.Collections.Generic;

////TODO: support ProcessEventsManually
////TODO: add way to pick by player index

// Some fields assigned through only through serialization.
#pragma warning disable CS0649

namespace UnityEngine.InputSystem.Samples
{
    /// <summary>
    /// A component for debugging purposes that adds an on-screen display which shows
    /// activity on an input action over time (<see cref="InputActionVisualizer.Visualization.Interaction"/>)
    /// or an action's current value (<see cref="InputActionVisualizer.Visualization.Value"/>).
    /// </summary>
    /// <seealso cref="InputControlVisualizer"/>
    [AddComponentMenu("Input/Debug/Input Action Visualizer")]
    [ExecuteInEditMode]
    public class InputActionVisualizer : InputVisualizer
    {
        /// <summary>
        /// The action that is being visualized. May be null.
        /// </summary>
        public InputAction action => m_Action;

        protected void FixedUpdate()
        {
            if (m_Visualization != Visualization.Value || m_Action == null || m_Visualizer == null)
                return;
            if (InputSystem.settings.updateMode != InputSettings.UpdateMode.ProcessEventsInFixedUpdate)
                return;
            RecordValue(Time.fixedTime);
        }

        protected void Update()
        {
            if (m_Visualization != Visualization.Value || m_Action == null || m_Visualizer == null)
                return;
            if (InputSystem.settings.updateMode != InputSettings.UpdateMode.ProcessEventsInDynamicUpdate)
                return;
            RecordValue(Time.time);
        }

        protected new void OnEnable()
        {
            if (m_Visualization == Visualization.None)
                return;

            base.OnEnable();

            ResolveAction();
            SetupVisualizer();

            if (s_EnabledInstances == null)
                s_EnabledInstances = new List<InputActionVisualizer>();
            if (s_EnabledInstances.Count == 0)
                InputSystem.onActionChange += OnActionChange;
            s_EnabledInstances.Add(this);
        }

        protected new void OnDisable()
        {
            base.OnDisable();

            s_EnabledInstances.Remove(this);
            if (s_EnabledInstances.Count == 0)
                InputSystem.onActionChange -= OnActionChange;

            if (m_Visualization == Visualization.Interaction && m_Action != null)
            {
                m_Action.started -= OnActionTriggered;
                m_Action.performed -= OnActionTriggered;
                m_Action.canceled -= OnActionTriggered;
            }
        }

        protected new void OnGUI()
        {
            if (m_Visualization == Visualization.None)
                return;

            if (Event.current.type != EventType.Repaint)
                return;

            base.OnGUI();

            if (m_ShowControlName && m_ActiveControlName != null)
                VisualizationHelpers.DrawText(m_ActiveControlName, new Vector2(m_Rect.x, m_Rect.yMax),
                    VisualizationHelpers.ValueTextStyle);
        }

        private void RecordValue(double time)
        {
            Debug.Assert(m_Action != null);
            Debug.Assert(m_Visualizer != null);

            var value = m_Action.ReadValueAsObject();
            m_Visualizer.AddSample(value, time);

            if (m_ShowControlName)
                RecordControlName();
        }

        private void RecordControlName()
        {
            var control = m_Action.activeControl;
            if (control == m_ActiveControl)
                return;

            m_ActiveControl = control;
            m_ActiveControlName = control != null ? new GUIContent(control.path) : null;
        }

        private void ResolveAction()
        {
            // If we have a reference to an action, try that first.
            if (m_ActionReference != null)
                m_Action = m_ActionReference.action;

            // If we didn't get an action from that but we have an action name,
            // just search through the currently enabled actions for one that
            // matches by name.
            if (m_Action == null && !string.IsNullOrEmpty(m_ActionName))
            {
                var slashIndex = m_ActionName.IndexOf('/');
                var mapName = slashIndex != -1 ? m_ActionName.Substring(0, slashIndex) : null;
                var actionName = slashIndex != -1 ? m_ActionName.Substring(slashIndex + 1) : m_ActionName;

                var enabledActions = InputSystem.ListEnabledActions();
                foreach (var action in enabledActions)
                {
                    if (string.Compare(actionName, action.name, StringComparison.InvariantCultureIgnoreCase) != 0)
                        continue;

                    if (mapName != null && action.actionMap != null && string.Compare(mapName, action.actionMap.name,
                        StringComparison.InvariantCultureIgnoreCase) != 0)
                        continue;

                    m_Action = action;
                    break;
                }
            }

            // If we still don't have an action, there's nothing much for us to do.
            // The action may show up at a later point.
            if (m_Action == null)
                return;

            if (m_Visualization == Visualization.Interaction)
            {
                m_Action.performed += OnActionTriggered;
                m_Action.started += OnActionTriggered;
                m_Action.canceled += OnActionTriggered;
            }
        }

        private void SetupVisualizer()
        {
            m_Visualizer = null;
            if (m_Action == null)
                return;

            switch (m_Visualization)
            {
                case Visualization.Value:
                    switch (m_Action.type)
                    {
                        case InputActionType.Button:
                            m_Visualizer = new VisualizationHelpers.ScalarVisualizer<float>
                            {
                                limitMax = 1
                            };
                            break;

                        case InputActionType.Value:
                        case InputActionType.PassThrough:
                            if (!string.IsNullOrEmpty(m_Action.expectedControlType))
                            {
                                var layout = InputSystem.LoadLayout(m_Action.expectedControlType);
                                if (layout != null)
                                {
                                    var valueType = layout.GetValueType();
                                    if (valueType == typeof(float))
                                        m_Visualizer = new VisualizationHelpers.ScalarVisualizer<float>
                                        {
                                            limitMax = 1
                                        };
                                    else if (valueType == typeof(int))
                                        m_Visualizer = new VisualizationHelpers.ScalarVisualizer<int>
                                        {
                                            limitMax = 1
                                        };
                                    else if (valueType == typeof(Vector2))
                                        m_Visualizer = new VisualizationHelpers.Vector2Visualizer();
                                }
                            }
                            break;
                    }
                    break;

                case Visualization.Interaction:
                    // We don't really know which interactions are sitting on the action and its bindings
                    // and while we could do and perform work to find out, it's simpler to just wait until
                    // we get input and then whatever interactions we encounter as we go along. Also keeps
                    // the visualization a little less cluttered.
                    m_Visualizer = new VisualizationHelpers.TimelineVisualizer();
                    break;
            }
        }

        private void OnActionDisabled()
        {
        }

        private void OnActionTriggered(InputAction.CallbackContext context)
        {
            Debug.Assert(m_Visualization == Visualization.Interaction);

            var timelineName = "Default";
            var interaction = context.interaction;
            if (interaction != null)
            {
                timelineName = interaction.GetType().Name;
                if (timelineName.EndsWith("Interaction"))
                    timelineName = timelineName.Substring(0, timelineName.Length - "Interaction".Length);
            }

            var visualizer = (VisualizationHelpers.TimelineVisualizer)m_Visualizer;
            var timelineIndex = visualizer.GetTimeline(timelineName);
            if (timelineIndex == -1)
            {
                Color color;
                timelineIndex = visualizer.timelineCount;
                if (timelineIndex < s_InteractionColors.Length)
                    color = s_InteractionColors[timelineIndex];
                else
                    color = new Color(Random.value, Random.value, Random.value, 1);

                visualizer.AddTimeline(timelineName, color);
                if (timelineIndex > 0)
                    visualizer.showLegend = true;
            }

            var time = (float)context.time;
            switch (context.phase)
            {
                case InputActionPhase.Canceled:
                    visualizer.AddSample(timelineIndex, 0f, time);
                    break;

                case InputActionPhase.Performed:
                    visualizer.AddSample(timelineIndex, 1f, time);
                    visualizer.AddSample(timelineIndex, 0f, time);
                    break;

                case InputActionPhase.Started:
                    visualizer.AddSample(timelineIndex, 0.5f, time);
                    break;
            }

            if (m_ShowControlName)
                RecordControlName();
        }

        private static void OnActionChange(object actionOrMap, InputActionChange change)
        {
            switch (change)
            {
                case InputActionChange.ActionEnabled:
                case InputActionChange.ActionMapEnabled:
                    for (var i = 0; i < s_EnabledInstances.Count; ++i)
                        if (s_EnabledInstances[i].m_Action == null)
                        {
                            s_EnabledInstances[i].ResolveAction();
                            if (s_EnabledInstances[i].m_Action != null)
                                s_EnabledInstances[i].SetupVisualizer();
                        }
                    break;

                case InputActionChange.ActionDisabled:
                    for (var i = 0; i < s_EnabledInstances.Count; ++i)
                        if (actionOrMap == s_EnabledInstances[i].m_Action)
                            s_EnabledInstances[i].OnActionDisabled();
                    break;

                case InputActionChange.ActionMapDisabled:
                    for (var i = 0; i < s_EnabledInstances.Count; ++i)
                        if (s_EnabledInstances[i].m_Action?.actionMap == actionOrMap)
                            s_EnabledInstances[i].OnActionDisabled();
                    break;
            }
        }

        [SerializeField] private Visualization m_Visualization;
        [SerializeField] private InputActionReference m_ActionReference;
        [SerializeField] private string m_ActionName;
        [SerializeField] private bool m_ShowControlName;

        [NonSerialized] private InputAction m_Action;
        [NonSerialized] private InputControl m_ActiveControl;
        [NonSerialized] private GUIContent m_ActiveControlName;

        private static List<InputActionVisualizer> s_EnabledInstances;
        private static readonly Color[] s_InteractionColors =
        {
            new Color(1, 0, 0, 1),
            new Color(0, 0, 1, 1),
            new Color(1, 1, 0, 1),
            new Color(1, 0, 1, 1),
            new Color(0, 1, 1, 1),
            new Color(0, 1, 0, 1),
        };

        public enum Visualization
        {
            None,
            Value,
            Interaction,
        }
    }
}
