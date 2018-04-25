using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.PostProcessing
{
    public sealed class CurveEditor
    {
        #region Enums

        enum EditMode
        {
            None,
            Moving,
            TangentEdit
        }

        enum Tangent
        {
            In,
            Out
        }
        #endregion

        #region Structs
        public struct Settings
        {
            public Rect bounds;
            public RectOffset padding;
            public Color selectionColor;
            public float curvePickingDistance;
            public float keyTimeClampingDistance;

            public static Settings defaultSettings
            {
                get
                {
                    return new Settings
                    {
                        bounds = new Rect(0f, 0f, 1f, 1f),
                        padding = new RectOffset(10, 10, 10, 10),
                        selectionColor = Color.yellow,
                        curvePickingDistance = 6f,
                        keyTimeClampingDistance = 1e-4f
                    };
                }
            }
        }

        public struct CurveState
        {
            public bool visible;
            public bool editable;
            public uint minPointCount;
            public float zeroKeyConstantValue;
            public Color color;
            public float width;
            public float handleWidth;
            public bool showNonEditableHandles;
            public bool onlyShowHandlesOnSelection;
            public bool loopInBounds;

            public static CurveState defaultState
            {
                get
                {
                    return new CurveState
                    {
                        visible = true,
                        editable = true,
                        minPointCount = 2,
                        zeroKeyConstantValue = 0f,
                        color = Color.white,
                        width = 2f,
                        handleWidth = 2f,
                        showNonEditableHandles = true,
                        onlyShowHandlesOnSelection = false,
                        loopInBounds = false
                    };
                }
            }
        }

        public struct Selection
        {
            public SerializedProperty curve;
            public int keyframeIndex;
            public Keyframe? keyframe;

            public Selection(SerializedProperty curve, int keyframeIndex, Keyframe? keyframe)
            {
                this.curve = curve;
                this.keyframeIndex = keyframeIndex;
                this.keyframe = keyframe;
            }
        }

        internal struct MenuAction
        {
            internal SerializedProperty curve;
            internal int index;
            internal Vector3 position;

            internal MenuAction(SerializedProperty curve)
            {
                this.curve = curve;
                this.index = -1;
                this.position = Vector3.zero;
            }

            internal MenuAction(SerializedProperty curve, int index)
            {
                this.curve = curve;
                this.index = index;
                this.position = Vector3.zero;
            }

            internal MenuAction(SerializedProperty curve, Vector3 position)
            {
                this.curve = curve;
                this.index = -1;
                this.position = position;
            }
        }
        #endregion

        #region Fields & properties
        public Settings settings { get; private set; }

        Dictionary<SerializedProperty, CurveState> m_Curves;
        Rect m_CurveArea;

        SerializedProperty m_SelectedCurve;
        int m_SelectedKeyframeIndex = -1;

        EditMode m_EditMode = EditMode.None;
        Tangent m_TangentEditMode;

        bool m_Dirty;
        #endregion

        #region Constructors & destructors
        public CurveEditor()
            : this(Settings.defaultSettings)
        {}

        public CurveEditor(Settings settings)
        {
            this.settings = settings;
            m_Curves = new Dictionary<SerializedProperty, CurveState>();
        }

        #endregion

        #region Public API
        public void Add(params SerializedProperty[] curves)
        {
            foreach (var curve in curves)
                Add(curve, CurveState.defaultState);
        }

        public void Add(SerializedProperty curve)
        {
            Add(curve, CurveState.defaultState);
        }

        public void Add(SerializedProperty curve, CurveState state)
        {
            // Make sure the property is in fact an AnimationCurve
            var animCurve = curve.animationCurveValue;
            if (animCurve == null)
                throw new ArgumentException("curve");

            if (m_Curves.ContainsKey(curve))
                Debug.LogWarning("Curve has already been added to the editor");

            m_Curves.Add(curve, state);
        }

        public void Remove(SerializedProperty curve)
        {
            m_Curves.Remove(curve);
        }

        public void RemoveAll()
        {
            m_Curves.Clear();
        }

        public CurveState GetCurveState(SerializedProperty curve)
        {
            CurveState state;
            if (!m_Curves.TryGetValue(curve, out state))
                throw new KeyNotFoundException("curve");

            return state;
        }

        public void SetCurveState(SerializedProperty curve, CurveState state)
        {
            if (!m_Curves.ContainsKey(curve))
                throw new KeyNotFoundException("curve");

            m_Curves[curve] = state;
        }

        public Selection GetSelection()
        {
            Keyframe? key = null;
            if (m_SelectedKeyframeIndex > -1)
            {
                var curve = m_SelectedCurve.animationCurveValue;

                if (m_SelectedKeyframeIndex >= curve.length)
                    m_SelectedKeyframeIndex = -1;
                else
                    key = curve[m_SelectedKeyframeIndex];
            }

            return new Selection(m_SelectedCurve, m_SelectedKeyframeIndex, key);
        }

        public void SetKeyframe(SerializedProperty curve, int keyframeIndex, Keyframe keyframe)
        {
            var animCurve = curve.animationCurveValue;
            SetKeyframe(animCurve, keyframeIndex, keyframe);
            SaveCurve(curve, animCurve);
        }

        public bool OnGUI(Rect rect)
        {
            if (Event.current.type == EventType.Repaint)
                m_Dirty = false;

            GUI.BeginClip(rect);
            {
                var area = new Rect(Vector2.zero, rect.size);
                m_CurveArea = settings.padding.Remove(area);

                foreach (var curve in m_Curves)
                    OnCurveGUI(area, curve.Key, curve.Value);

                OnGeneralUI(area);
            }
            GUI.EndClip();

            return m_Dirty;
        }

        #endregion

        #region UI & events

        void OnCurveGUI(Rect rect, SerializedProperty curve, CurveState state)
        {
            // Discard invisible curves
            if (!state.visible)
                return;

            var animCurve = curve.animationCurveValue;
            var keys = animCurve.keys;
            var length = keys.Length;

            // Curve drawing
            // Slightly dim non-editable curves
            var color = state.color;
            if (!state.editable)
                color.a *= 0.5f;

            Handles.color = color;
            var bounds = settings.bounds;

            if (length == 0)
            {
                var p1 = CurveToCanvas(new Vector3(bounds.xMin, state.zeroKeyConstantValue));
                var p2 = CurveToCanvas(new Vector3(bounds.xMax, state.zeroKeyConstantValue));
                Handles.DrawAAPolyLine(state.width, p1, p2);
            }
            else if (length == 1)
            {
                var p1 = CurveToCanvas(new Vector3(bounds.xMin, keys[0].value));
                var p2 = CurveToCanvas(new Vector3(bounds.xMax, keys[0].value));
                Handles.DrawAAPolyLine(state.width, p1, p2);
            }
            else
            {
                var prevKey = keys[0];
                for (int k = 1; k < length; k++)
                {
                    var key = keys[k];
                    var pts = BezierSegment(prevKey, key);

                    if (float.IsInfinity(prevKey.outTangent) || float.IsInfinity(key.inTangent))
                    {
                        var s = HardSegment(prevKey, key);
                        Handles.DrawAAPolyLine(state.width, s[0], s[1], s[2]);
                    }
                    else Handles.DrawBezier(pts[0], pts[3], pts[1], pts[2], color, null, state.width);

                    prevKey = key;
                }

                // Curve extents & loops
                if (keys[0].time > bounds.xMin)
                {
                    if (state.loopInBounds)
                    {
                        var p1 = keys[length - 1];
                        p1.time -= settings.bounds.width;
                        var p2 = keys[0];
                        var pts = BezierSegment(p1, p2);

                        if (float.IsInfinity(p1.outTangent) || float.IsInfinity(p2.inTangent))
                        {
                            var s = HardSegment(p1, p2);
                            Handles.DrawAAPolyLine(state.width, s[0], s[1], s[2]);
                        }
                        else Handles.DrawBezier(pts[0], pts[3], pts[1], pts[2], color, null, state.width);
                    }
                    else
                    {
                        var p1 = CurveToCanvas(new Vector3(bounds.xMin, keys[0].value));
                        var p2 = CurveToCanvas(keys[0]);
                        Handles.DrawAAPolyLine(state.width, p1, p2);
                    }
                }

                if (keys[length - 1].time < bounds.xMax)
                {
                    if (state.loopInBounds)
                    {
                        var p1 = keys[length - 1];
                        var p2 = keys[0];
                        p2.time += settings.bounds.width;
                        var pts = BezierSegment(p1, p2);

                        if (float.IsInfinity(p1.outTangent) || float.IsInfinity(p2.inTangent))
                        {
                            var s = HardSegment(p1, p2);
                            Handles.DrawAAPolyLine(state.width, s[0], s[1], s[2]);
                        }
                        else Handles.DrawBezier(pts[0], pts[3], pts[1], pts[2], color, null, state.width);
                    }
                    else
                    {
                        var p1 = CurveToCanvas(keys[length - 1]);
                        var p2 = CurveToCanvas(new Vector3(bounds.xMax, keys[length - 1].value));
                        Handles.DrawAAPolyLine(state.width, p1, p2);
                    }
                }
            }

            // Make sure selection is correct (undo can break it)
            bool isCurrentlySelectedCurve = curve == m_SelectedCurve;

            if (isCurrentlySelectedCurve && m_SelectedKeyframeIndex >= length)
                m_SelectedKeyframeIndex = -1;

            // Handles & keys
            for (int k = 0; k < length; k++)
            {
                bool isCurrentlySelectedKeyframe = k == m_SelectedKeyframeIndex;
                var e = Event.current;

                var pos = CurveToCanvas(keys[k]);
                var hitRect = new Rect(pos.x - 8f, pos.y - 8f, 16f, 16f);
                var offset = isCurrentlySelectedCurve
                    ? new RectOffset(5, 5, 5, 5)
                    : new RectOffset(6, 6, 6, 6);

                var outTangent = pos + CurveTangentToCanvas(keys[k].outTangent).normalized * 40f;
                var inTangent = pos - CurveTangentToCanvas(keys[k].inTangent).normalized * 40f;
                var inTangentHitRect = new Rect(inTangent.x - 7f, inTangent.y - 7f, 14f, 14f);
                var outTangentHitrect = new Rect(outTangent.x - 7f, outTangent.y - 7f, 14f, 14f);

                // Draw
                if (state.showNonEditableHandles)
                {
                    if (e.type == EventType.Repaint)
                    {
                        var selectedColor = (isCurrentlySelectedCurve && isCurrentlySelectedKeyframe)
                            ? settings.selectionColor
                            : state.color;

                        // Keyframe
                        EditorGUI.DrawRect(offset.Remove(hitRect), selectedColor);

                        // Tangents
                        if (isCurrentlySelectedCurve && (!state.onlyShowHandlesOnSelection || (state.onlyShowHandlesOnSelection && isCurrentlySelectedKeyframe)))
                        {
                            Handles.color = selectedColor;

                            if (k > 0 || state.loopInBounds)
                            {
                                Handles.DrawAAPolyLine(state.handleWidth, pos, inTangent);
                                EditorGUI.DrawRect(offset.Remove(inTangentHitRect), selectedColor);
                            }

                            if (k < length - 1 || state.loopInBounds)
                            {
                                Handles.DrawAAPolyLine(state.handleWidth, pos, outTangent);
                                EditorGUI.DrawRect(offset.Remove(outTangentHitrect), selectedColor);
                            }
                        }
                    }
                }

                // Events
                if (state.editable)
                {
                    // Keyframe move
                    if (m_EditMode == EditMode.Moving && e.type == EventType.MouseDrag && isCurrentlySelectedCurve && isCurrentlySelectedKeyframe)
                    {
                        EditMoveKeyframe(animCurve, keys, k);
                    }

                    // Tangent editing
                    if (m_EditMode == EditMode.TangentEdit && e.type == EventType.MouseDrag && isCurrentlySelectedCurve && isCurrentlySelectedKeyframe)
                    {
                        bool alreadyBroken = !(Mathf.Approximately(keys[k].inTangent, keys[k].outTangent) || (float.IsInfinity(keys[k].inTangent) && float.IsInfinity(keys[k].outTangent)));
                        EditMoveTangent(animCurve, keys, k, m_TangentEditMode, e.shift || !(alreadyBroken || e.control));
                    }

                    // Keyframe selection & context menu
                    if (e.type == EventType.MouseDown && rect.Contains(e.mousePosition))
                    {
                        if (hitRect.Contains(e.mousePosition))
                        {
                            if (e.button == 0)
                            {
                                SelectKeyframe(curve, k);
                                m_EditMode = EditMode.Moving;
                                e.Use();
                            }
                            else if (e.button == 1)
                            {
                                // Keyframe context menu
                                var menu = new GenericMenu();
                                menu.AddItem(new GUIContent("Delete Key"), false, (x) =>
                                {
                                    var action = (MenuAction)x;
                                    var curveValue = action.curve.animationCurveValue;
                                    action.curve.serializedObject.Update();
                                    RemoveKeyframe(curveValue, action.index);
                                    m_SelectedKeyframeIndex = -1;
                                    SaveCurve(action.curve, curveValue);
                                    action.curve.serializedObject.ApplyModifiedProperties();
                                }, new MenuAction(curve, k));
                                menu.ShowAsContext();
                                e.Use();
                            }
                        }
                    }

                    // Tangent selection & edit mode
                    if (e.type == EventType.MouseDown && rect.Contains(e.mousePosition))
                    {
                        if (inTangentHitRect.Contains(e.mousePosition) && (k > 0 || state.loopInBounds))
                        {
                            SelectKeyframe(curve, k);
                            m_EditMode = EditMode.TangentEdit;
                            m_TangentEditMode = Tangent.In;
                            e.Use();
                        }
                        else if (outTangentHitrect.Contains(e.mousePosition) && (k < length - 1 || state.loopInBounds))
                        {
                            SelectKeyframe(curve, k);
                            m_EditMode = EditMode.TangentEdit;
                            m_TangentEditMode = Tangent.Out;
                            e.Use();
                        }
                    }

                    // Mouse up - clean up states
                    if (e.rawType == EventType.MouseUp && m_EditMode != EditMode.None)
                    {
                        m_EditMode = EditMode.None;
                    }

                    // Set cursors
                    {
                        EditorGUIUtility.AddCursorRect(hitRect, MouseCursor.MoveArrow);

                        if (k > 0 || state.loopInBounds)
                            EditorGUIUtility.AddCursorRect(inTangentHitRect, MouseCursor.RotateArrow);

                        if (k < length - 1 || state.loopInBounds)
                            EditorGUIUtility.AddCursorRect(outTangentHitrect, MouseCursor.RotateArrow);
                    }
                }
            }

            Handles.color = Color.white;
            SaveCurve(curve, animCurve);
        }

        void OnGeneralUI(Rect rect)
        {
            var e = Event.current;

            // Selection
            if (e.type == EventType.MouseDown)
            {
                GUI.FocusControl(null);
                m_SelectedCurve = null;
                m_SelectedKeyframeIndex = -1;
                bool used = false;

                var hit = CanvasToCurve(e.mousePosition);
                float curvePickValue = CurveToCanvas(hit).y;

                // Try and select a curve
                foreach (var curve in m_Curves)
                {
                    if (!curve.Value.editable || !curve.Value.visible)
                        continue;

                    var prop = curve.Key;
                    var state = curve.Value;
                    var animCurve = prop.animationCurveValue;
                    float hitY = animCurve.length == 0
                        ? state.zeroKeyConstantValue
                        : animCurve.Evaluate(hit.x);

                    var curvePos = CurveToCanvas(new Vector3(hit.x, hitY));

                    if (Mathf.Abs(curvePos.y - curvePickValue) < settings.curvePickingDistance)
                    {
                        m_SelectedCurve = prop;

                        if (e.clickCount == 2 && e.button == 0)
                        {
                            // Create a keyframe on double-click on this curve
                            EditCreateKeyframe(animCurve, hit, true, state.zeroKeyConstantValue);
                            SaveCurve(prop, animCurve);
                        }
                        else if (e.button == 1)
                        {
                            // Curve context menu
                            var menu = new GenericMenu();
                            menu.AddItem(new GUIContent("Add Key"), false, (x) =>
                            {
                                var action = (MenuAction)x;
                                var curveValue = action.curve.animationCurveValue;
                                action.curve.serializedObject.Update();
                                EditCreateKeyframe(curveValue, hit, true, 0f);
                                SaveCurve(action.curve, curveValue);
                                action.curve.serializedObject.ApplyModifiedProperties();
                            }, new MenuAction(prop, hit));
                            menu.ShowAsContext();
                            e.Use();
                            used = true;
                        }
                    }
                }

                if (e.clickCount == 2 && e.button == 0 && m_SelectedCurve == null)
                {
                    // Create a keyframe on every curve on double-click
                    foreach (var curve in m_Curves)
                    {
                        if (!curve.Value.editable || !curve.Value.visible)
                            continue;

                        var prop = curve.Key;
                        var state = curve.Value;
                        var animCurve = prop.animationCurveValue;
                        EditCreateKeyframe(animCurve, hit, e.alt, state.zeroKeyConstantValue);
                        SaveCurve(prop, animCurve);
                    }
                }
                else if (!used && e.button == 1)
                {
                    // Global context menu
                    var menu = new GenericMenu();
                    menu.AddItem(new GUIContent("Add Key At Position"), false, () => ContextMenuAddKey(hit, false));
                    menu.AddItem(new GUIContent("Add Key On Curves"), false, () => ContextMenuAddKey(hit, true));
                    menu.ShowAsContext();
                }

                e.Use();
            }

            // Delete selected key(s)
            if (e.type == EventType.KeyDown && (e.keyCode == KeyCode.Delete || e.keyCode == KeyCode.Backspace))
            {
                if (m_SelectedKeyframeIndex != -1 && m_SelectedCurve != null)
                {
                    var animCurve = m_SelectedCurve.animationCurveValue;
                    var length = animCurve.length;

                    if (m_Curves[m_SelectedCurve].minPointCount < length && length >= 0)
                    {
                        EditDeleteKeyframe(animCurve, m_SelectedKeyframeIndex);
                        m_SelectedKeyframeIndex = -1;
                        SaveCurve(m_SelectedCurve, animCurve);
                    }

                    e.Use();
                }
            }
        }

        void SaveCurve(SerializedProperty prop, AnimationCurve curve)
        {
            prop.animationCurveValue = curve;
        }

        void Invalidate()
        {
            m_Dirty = true;
        }

        #endregion

        #region Keyframe manipulations

        void SelectKeyframe(SerializedProperty curve, int keyframeIndex)
        {
            m_SelectedKeyframeIndex = keyframeIndex;
            m_SelectedCurve = curve;
            Invalidate();
        }

        void ContextMenuAddKey(Vector3 hit, bool createOnCurve)
        {
            SerializedObject serializedObject = null;

            foreach (var curve in m_Curves)
            {
                if (!curve.Value.editable || !curve.Value.visible)
                    continue;

                var prop = curve.Key;
                var state = curve.Value;

                if (serializedObject == null)
                {
                    serializedObject = prop.serializedObject;
                    serializedObject.Update();
                }

                var animCurve = prop.animationCurveValue;
                EditCreateKeyframe(animCurve, hit, createOnCurve, state.zeroKeyConstantValue);
                SaveCurve(prop, animCurve);
            }

            if (serializedObject != null)
                serializedObject.ApplyModifiedProperties();

            Invalidate();
        }

        void EditCreateKeyframe(AnimationCurve curve, Vector3 position, bool createOnCurve, float zeroKeyConstantValue)
        {
            float tangent = EvaluateTangent(curve, position.x);

            if (createOnCurve)
            {
                position.y = curve.length == 0
                    ? zeroKeyConstantValue
                    : curve.Evaluate(position.x);
            }

            AddKeyframe(curve, new Keyframe(position.x, position.y, tangent, tangent));
        }

        void EditDeleteKeyframe(AnimationCurve curve, int keyframeIndex)
        {
            RemoveKeyframe(curve, keyframeIndex);
        }

        void AddKeyframe(AnimationCurve curve, Keyframe newValue)
        {
            curve.AddKey(newValue);
            Invalidate();
        }

        void RemoveKeyframe(AnimationCurve curve, int keyframeIndex)
        {
            curve.RemoveKey(keyframeIndex);
            Invalidate();
        }

        void SetKeyframe(AnimationCurve curve, int keyframeIndex, Keyframe newValue)
        {
            var keys = curve.keys;

            if (keyframeIndex > 0)
                newValue.time = Mathf.Max(keys[keyframeIndex - 1].time + settings.keyTimeClampingDistance, newValue.time);

            if (keyframeIndex < keys.Length - 1)
                newValue.time = Mathf.Min(keys[keyframeIndex + 1].time - settings.keyTimeClampingDistance, newValue.time);

            curve.MoveKey(keyframeIndex, newValue);
            Invalidate();
        }

        void EditMoveKeyframe(AnimationCurve curve, Keyframe[] keys, int keyframeIndex)
        {
            var key = CanvasToCurve(Event.current.mousePosition);
            float inTgt = keys[keyframeIndex].inTangent;
            float outTgt = keys[keyframeIndex].outTangent;
            SetKeyframe(curve, keyframeIndex, new Keyframe(key.x, key.y, inTgt, outTgt));
        }

        void EditMoveTangent(AnimationCurve curve, Keyframe[] keys, int keyframeIndex, Tangent targetTangent, bool linkTangents)
        {
            var pos = CanvasToCurve(Event.current.mousePosition);

            float time = keys[keyframeIndex].time;
            float value = keys[keyframeIndex].value;

            pos -= new Vector3(time, value);

            if (targetTangent == Tangent.In && pos.x > 0f)
                pos.x = 0f;

            if (targetTangent == Tangent.Out && pos.x < 0f)
                pos.x = 0f;

            float tangent;

            if (Mathf.Approximately(pos.x, 0f))
                tangent = pos.y < 0f ? float.PositiveInfinity : float.NegativeInfinity;
            else
                tangent = pos.y / pos.x;

            float inTangent = keys[keyframeIndex].inTangent;
            float outTangent = keys[keyframeIndex].outTangent;

            if (targetTangent == Tangent.In || linkTangents)
                inTangent = tangent;
            if (targetTangent == Tangent.Out || linkTangents)
                outTangent = tangent;

            SetKeyframe(curve, keyframeIndex, new Keyframe(time, value, inTangent, outTangent));
        }

        #endregion

        #region Maths utilities

        Vector3 CurveToCanvas(Keyframe keyframe)
        {
            return CurveToCanvas(new Vector3(keyframe.time, keyframe.value));
        }

        Vector3 CurveToCanvas(Vector3 position)
        {
            var bounds = settings.bounds;
            var output = new Vector3((position.x - bounds.x) / (bounds.xMax - bounds.x), (position.y - bounds.y) / (bounds.yMax - bounds.y));
            output.x = output.x * (m_CurveArea.xMax - m_CurveArea.xMin) + m_CurveArea.xMin;
            output.y = (1f - output.y) * (m_CurveArea.yMax - m_CurveArea.yMin) + m_CurveArea.yMin;
            return output;
        }

        Vector3 CanvasToCurve(Vector3 position)
        {
            var bounds = settings.bounds;
            var output = position;
            output.x = (output.x - m_CurveArea.xMin) / (m_CurveArea.xMax - m_CurveArea.xMin);
            output.y = (output.y - m_CurveArea.yMin) / (m_CurveArea.yMax - m_CurveArea.yMin);
            output.x = Mathf.Lerp(bounds.x, bounds.xMax, output.x);
            output.y = Mathf.Lerp(bounds.yMax, bounds.y, output.y);
            return output;
        }

        Vector3 CurveTangentToCanvas(float tangent)
        {
            if (!float.IsInfinity(tangent))
            {
                var bounds = settings.bounds;
                float ratio = (m_CurveArea.width / m_CurveArea.height) / ((bounds.xMax - bounds.x) / (bounds.yMax - bounds.y));
                return new Vector3(1f, -tangent / ratio).normalized;
            }

            return float.IsPositiveInfinity(tangent) ? Vector3.up : Vector3.down;
        }

        Vector3[] BezierSegment(Keyframe start, Keyframe end)
        {
            var segment = new Vector3[4];

            segment[0] = CurveToCanvas(new Vector3(start.time, start.value));
            segment[3] = CurveToCanvas(new Vector3(end.time, end.value));

            float middle  = start.time + ((end.time - start.time) * 0.333333f);
            float middle2 = start.time + ((end.time - start.time) * 0.666666f);

            segment[1] = CurveToCanvas(new Vector3(middle, ProjectTangent(start.time, start.value, start.outTangent, middle)));
            segment[2] = CurveToCanvas(new Vector3(middle2, ProjectTangent(end.time, end.value, end.inTangent, middle2)));

            return segment;
        }

        Vector3[] HardSegment(Keyframe start, Keyframe end)
        {
            var segment = new Vector3[3];

            segment[0] = CurveToCanvas(start);
            segment[1] = CurveToCanvas(new Vector3(end.time, start.value));
            segment[2] = CurveToCanvas(end);

            return segment;
        }

        float ProjectTangent(float inPosition, float inValue, float inTangent, float projPosition)
        {
            return inValue + ((projPosition - inPosition) * inTangent);
        }

        float EvaluateTangent(AnimationCurve curve, float time)
        {
            int prev = -1, next = 0;
            for (int i = 0; i < curve.keys.Length; i++)
            {
                if (time > curve.keys[i].time)
                {
                    prev = i;
                    next = i + 1;
                }
                else break;
            }

            if (next == 0)
                return 0f;

            if (prev == curve.keys.Length - 1)
                return 0f;

            const float kD = 1e-3f;
            float tp = Mathf.Max(time - kD, curve.keys[prev].time);
            float tn = Mathf.Min(time + kD, curve.keys[next].time);

            float vp = curve.Evaluate(tp);
            float vn = curve.Evaluate(tn);

            if (Mathf.Approximately(tn, tp))
                return (vn - vp > 0f) ? float.PositiveInfinity : float.NegativeInfinity;

            return (vn - vp) / (tn - tp);
        }

        #endregion
    }
}
