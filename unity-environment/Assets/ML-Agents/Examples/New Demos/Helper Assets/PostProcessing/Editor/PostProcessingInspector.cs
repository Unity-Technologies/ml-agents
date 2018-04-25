using UnityEngine;
using UnityEngine.PostProcessing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace UnityEditor.PostProcessing
{
    //[CanEditMultipleObjects]
    [CustomEditor(typeof(PostProcessingProfile))]
    public class PostProcessingInspector : Editor
    {
        static GUIContent s_PreviewTitle = new GUIContent("Monitors");

        PostProcessingProfile m_ConcreteTarget
        {
            get { return target as PostProcessingProfile; }
        }

        int m_CurrentMonitorID
        {
            get { return m_ConcreteTarget.monitors.currentMonitorID; }
            set { m_ConcreteTarget.monitors.currentMonitorID = value; }
        }

        List<PostProcessingMonitor> m_Monitors;
        GUIContent[] m_MonitorNames;
        Dictionary<PostProcessingModelEditor, PostProcessingModel> m_CustomEditors = new Dictionary<PostProcessingModelEditor, PostProcessingModel>();

        public bool IsInteractivePreviewOpened { get; private set; }

        void OnEnable()
        {
            if (target == null)
                return;

            // Aggregate custom post-fx editors
            var assembly = Assembly.GetAssembly(typeof(PostProcessingInspector));

            var editorTypes = assembly.GetTypes()
                .Where(x => x.IsDefined(typeof(PostProcessingModelEditorAttribute), false));

            var customEditors = new Dictionary<Type, PostProcessingModelEditor>();
            foreach (var editor in editorTypes)
            {
                var attr = (PostProcessingModelEditorAttribute)editor.GetCustomAttributes(typeof(PostProcessingModelEditorAttribute), false)[0];
                var effectType = attr.type;
                var alwaysEnabled = attr.alwaysEnabled;

                var editorInst = (PostProcessingModelEditor)Activator.CreateInstance(editor);
                editorInst.alwaysEnabled = alwaysEnabled;
                editorInst.profile = target as PostProcessingProfile;
                editorInst.inspector = this;
                customEditors.Add(effectType, editorInst);
            }

            // ... and corresponding models
            var baseType = target.GetType();
            var property = serializedObject.GetIterator();

            while (property.Next(true))
            {
                if (!property.hasChildren)
                    continue;

                var type = baseType;
                var srcObject = ReflectionUtils.GetFieldValueFromPath(serializedObject.targetObject, ref type, property.propertyPath);

                if (srcObject == null)
                    continue;

                PostProcessingModelEditor editor;
                if (customEditors.TryGetValue(type, out editor))
                {
                    var effect = (PostProcessingModel)srcObject;

                    if (editor.alwaysEnabled)
                        effect.enabled = editor.alwaysEnabled;

                    m_CustomEditors.Add(editor, effect);
                    editor.target = effect;
                    editor.serializedProperty = property.Copy();
                    editor.OnPreEnable();
                }
            }

            // Prepare monitors
            m_Monitors = new List<PostProcessingMonitor>();

            var monitors = new List<PostProcessingMonitor>
            {
                new HistogramMonitor(),
                new WaveformMonitor(),
                new ParadeMonitor(),
                new VectorscopeMonitor()
            };

            var monitorNames = new List<GUIContent>();

            foreach (var monitor in monitors)
            {
                if (monitor.IsSupported())
                {
                    monitor.Init(m_ConcreteTarget.monitors, this);
                    m_Monitors.Add(monitor);
                    monitorNames.Add(monitor.GetMonitorTitle());
                }
            }

            m_MonitorNames = monitorNames.ToArray();

            if (m_Monitors.Count > 0)
                m_ConcreteTarget.monitors.onFrameEndEditorOnly = OnFrameEnd;
        }

        void OnDisable()
        {
            if (m_CustomEditors != null)
            {
                foreach (var editor in m_CustomEditors.Keys)
                    editor.OnDisable();

                m_CustomEditors.Clear();
            }

            if (m_Monitors != null)
            {
                foreach (var monitor in m_Monitors)
                    monitor.Dispose();

                m_Monitors.Clear();
            }

            if (m_ConcreteTarget != null)
                m_ConcreteTarget.monitors.onFrameEndEditorOnly = null;
        }

        void OnFrameEnd(RenderTexture source)
        {
            if (!IsInteractivePreviewOpened)
                return;

            if (m_CurrentMonitorID < m_Monitors.Count)
                m_Monitors[m_CurrentMonitorID].OnFrameData(source);

            IsInteractivePreviewOpened = false;
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            // Handles undo/redo events first (before they get used by the editors' widgets)
            var e = Event.current;
            if (e.type == EventType.ValidateCommand && e.commandName == "UndoRedoPerformed")
            {
                foreach (var editor in m_CustomEditors)
                    editor.Value.OnValidate();
            }

            if (!m_ConcreteTarget.debugViews.IsModeActive(BuiltinDebugViewsModel.Mode.None))
                EditorGUILayout.HelpBox("A debug view is currently enabled. Changes done to an effect might not be visible.", MessageType.Info);

            foreach (var editor in m_CustomEditors)
            {
                EditorGUI.BeginChangeCheck();

                editor.Key.OnGUI();

                if (EditorGUI.EndChangeCheck())
                    editor.Value.OnValidate();
            }

            serializedObject.ApplyModifiedProperties();
        }

        public override GUIContent GetPreviewTitle()
        {
            return s_PreviewTitle;
        }

        public override bool HasPreviewGUI()
        {
            return GraphicsUtils.supportsDX11 && m_Monitors.Count > 0;
        }

        public override void OnPreviewSettings()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                if (m_CurrentMonitorID < m_Monitors.Count)
                    m_Monitors[m_CurrentMonitorID].OnMonitorSettings();

                GUILayout.Space(5);
                m_CurrentMonitorID = EditorGUILayout.Popup(m_CurrentMonitorID, m_MonitorNames, FxStyles.preDropdown, GUILayout.MaxWidth(100f));
            }
        }

        public override void OnInteractivePreviewGUI(Rect r, GUIStyle background)
        {
            IsInteractivePreviewOpened = true;

            if (m_CurrentMonitorID < m_Monitors.Count)
                m_Monitors[m_CurrentMonitorID].OnMonitorGUI(r);
        }
    }
}
