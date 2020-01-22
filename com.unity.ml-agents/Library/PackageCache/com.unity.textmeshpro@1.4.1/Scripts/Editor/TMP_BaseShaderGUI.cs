using UnityEngine;
using UnityEditor;

namespace TMPro.EditorUtilities
{
    /// <summary>Base class for TextMesh Pro shader GUIs.</summary>
    public abstract class TMP_BaseShaderGUI : ShaderGUI
    {
        /// <summary>Representation of a #pragma shader_feature.</summary>
        /// <description>It is assumed that the first feature option is for no keyword (underscores).</description>
        protected class ShaderFeature
        {
            public string undoLabel;

            public GUIContent label;

            /// <summary>The keyword labels, for display. Include the no-keyword as the first option.</summary>
            public GUIContent[] keywordLabels;

            /// <summary>The shader keywords. Exclude the no-keyword option.</summary>
            public string[] keywords;

            int m_State;

            public bool Active
            {
                get { return m_State >= 0; }
            }

            public int State
            {
                get { return m_State; }
            }

            public void ReadState(Material material)
            {
                for (int i = 0; i < keywords.Length; i++)
                {
                    if (material.IsKeywordEnabled(keywords[i]))
                    {
                        m_State = i;
                        return;
                    }
                }

                m_State = -1;
            }

            public void SetActive(bool active, Material material)
            {
                m_State = active ? 0 : -1;
                SetStateKeywords(material);
            }

            public void DoPopup(MaterialEditor editor, Material material)
            {
                EditorGUI.BeginChangeCheck();
                int selection = EditorGUILayout.Popup(label, m_State + 1, keywordLabels);
                if (EditorGUI.EndChangeCheck())
                {
                    m_State = selection - 1;
                    editor.RegisterPropertyChangeUndo(undoLabel);
                    SetStateKeywords(material);
                }
            }

            void SetStateKeywords(Material material)
            {
                for (int i = 0; i < keywords.Length; i++)
                {
                    if (i == m_State)
                    {
                        material.EnableKeyword(keywords[i]);
                    }
                    else
                    {
                        material.DisableKeyword(keywords[i]);
                    }
                }
            }
        }

        static GUIContent s_TempLabel = new GUIContent();

        protected static bool s_DebugExtended;

        static int s_UndoRedoCount, s_LastSeenUndoRedoCount;

        static float[][] s_TempFloats =
        {
            null, new float[1], new float[2], new float[3], new float[4]
        };

        protected static GUIContent[] s_XywhVectorLabels =
        {
            new GUIContent("X"),
            new GUIContent("Y"),
            new GUIContent("W", "Width"),
            new GUIContent("H", "Height")
        };

        protected static GUIContent[] s_LbrtVectorLabels =
        {
            new GUIContent("L", "Left"),
            new GUIContent("B", "Bottom"),
            new GUIContent("R", "Right"),
            new GUIContent("T", "Top")
        };

        static TMP_BaseShaderGUI()
        {
            // Keep track of how many undo/redo events happened.
            Undo.undoRedoPerformed += () => s_UndoRedoCount += 1;
        }

        bool m_IsNewGUI = true;

        float m_DragAndDropMinY;

        protected MaterialEditor m_Editor;

        protected Material m_Material;

        protected MaterialProperty[] m_Properties;

        void PrepareGUI()
        {
            m_IsNewGUI = false;
            ShaderUtilities.GetShaderPropertyIDs();

            // New GUI just got constructed. This happens in response to a selection,
            // but also after undo/redo events.
            if (s_LastSeenUndoRedoCount != s_UndoRedoCount)
            {
                // There's been at least one undo/redo since the last time this GUI got constructed.
                // Maybe the undo/redo was for this material? Assume that is was.
                TMPro_EventManager.ON_MATERIAL_PROPERTY_CHANGED(true, m_Material as Material);
            }

            s_LastSeenUndoRedoCount = s_UndoRedoCount;
        }

        public override void OnGUI(MaterialEditor materialEditor, MaterialProperty[] properties)
        {
            m_Editor = materialEditor;
            m_Material = materialEditor.target as Material;
            this.m_Properties = properties;

            if (m_IsNewGUI)
            {
                PrepareGUI();
            }

            DoDragAndDropBegin();
            EditorGUI.BeginChangeCheck();
            DoGUI();
            if (EditorGUI.EndChangeCheck())
            {
                TMPro_EventManager.ON_MATERIAL_PROPERTY_CHANGED(true, m_Material);
            }

            DoDragAndDropEnd();
        }

        /// <summary>Override this method to create the specific shader GUI.</summary>
        protected abstract void DoGUI();

        static string[] s_PanelStateLabel = new string[] { "\t- <i>Click to collapse</i> -", "\t- <i>Click to expand</i>  -" };

        protected bool BeginPanel(string panel, bool expanded)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            Rect r = EditorGUI.IndentedRect(GUILayoutUtility.GetRect(20, 18));
            r.x += 20;
            r.width += 6;

            bool enabled = GUI.enabled;
            GUI.enabled = true;
            expanded = TMP_EditorUtility.EditorToggle(r, expanded, new GUIContent(panel), TMP_UIStyleManager.panelTitle);
            r.width -= 30;
            EditorGUI.LabelField(r, new GUIContent(expanded ? s_PanelStateLabel[0] : s_PanelStateLabel[1]), TMP_UIStyleManager.rightLabel);
            GUI.enabled = enabled;

            EditorGUI.indentLevel += 1;
            EditorGUI.BeginDisabledGroup(false);

            return expanded;
        }

        protected bool BeginPanel(string panel, ShaderFeature feature, bool expanded, bool readState = true)
        {
            if (readState)
            {
                feature.ReadState(m_Material);
            }

            EditorGUI.BeginChangeCheck();

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            GUILayout.BeginHorizontal();

            Rect r = EditorGUI.IndentedRect(GUILayoutUtility.GetRect(20, 20, GUILayout.Width(20f)));
            bool active = EditorGUI.Toggle(r, feature.Active);

            if (EditorGUI.EndChangeCheck())
            {
                m_Editor.RegisterPropertyChangeUndo(feature.undoLabel);
                feature.SetActive(active, m_Material);
            }

            r = EditorGUI.IndentedRect(GUILayoutUtility.GetRect(20, 18));
            r.width += 6;

            bool enabled = GUI.enabled;
            GUI.enabled = true;
            expanded = TMP_EditorUtility.EditorToggle(r, expanded, new GUIContent(panel), TMP_UIStyleManager.panelTitle);
            r.width -= 10;
            EditorGUI.LabelField(r, new GUIContent(expanded ? s_PanelStateLabel[0] : s_PanelStateLabel[1]), TMP_UIStyleManager.rightLabel);
            GUI.enabled = enabled;

            GUILayout.EndHorizontal();

            EditorGUI.indentLevel += 1;
            EditorGUI.BeginDisabledGroup(!active);

            return expanded;
        }

        public void EndPanel()
        {
            EditorGUI.EndDisabledGroup();
            EditorGUI.indentLevel -= 1;
            EditorGUILayout.EndVertical();
        }

        MaterialProperty BeginProperty(string name)
        {
            MaterialProperty property = FindProperty(name, m_Properties);
            EditorGUI.BeginChangeCheck();
            EditorGUI.showMixedValue = property.hasMixedValue;
            m_Editor.BeginAnimatedCheck(Rect.zero, property);

            return property;
        }

        bool EndProperty()
        {
            m_Editor.EndAnimatedCheck();
            EditorGUI.showMixedValue = false;
            return EditorGUI.EndChangeCheck();
        }

        protected void DoPopup(string name, string label, GUIContent[] options)
        {
            MaterialProperty property = BeginProperty(name);
            s_TempLabel.text = label;
            int index = EditorGUILayout.Popup(s_TempLabel, (int)property.floatValue, options);
            if (EndProperty())
            {
                property.floatValue = index;
            }
        }

        protected void DoCubeMap(string name, string label)
        {
            DoTexture(name, label, typeof(Cubemap));
        }

        protected void DoTexture2D(string name, string label, bool withTilingOffset = false, string[] speedNames = null)
        {
            DoTexture(name, label, typeof(Texture2D), withTilingOffset, speedNames);
        }

        void DoTexture(string name, string label, System.Type type, bool withTilingOffset = false, string[] speedNames = null)
        {
            MaterialProperty property = BeginProperty(name);
            Rect rect = EditorGUILayout.GetControlRect(true, 60f);
            float totalWidth = rect.width;
            rect.width = EditorGUIUtility.labelWidth + 60f;
            s_TempLabel.text = label;
            Object tex = EditorGUI.ObjectField(rect, s_TempLabel, property.textureValue, type, false);

            if (EndProperty())
            {
                property.textureValue = tex as Texture;
            }

            rect.x += rect.width + 4f;
            rect.width = totalWidth - rect.width - 4f;
            rect.height = EditorGUIUtility.singleLineHeight;

            if (withTilingOffset)
            {
                DoTilingOffset(rect, property);
                rect.y += (rect.height + 2f) * 2f;
            }

            if (speedNames != null)
            {
                DoUVSpeed(rect, speedNames);
            }
        }

        void DoTilingOffset(Rect rect, MaterialProperty property)
        {
            float labelWidth = EditorGUIUtility.labelWidth;
            int indentLevel = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            EditorGUIUtility.labelWidth = Mathf.Min(40f, rect.width * 0.20f);

            Vector4 vector = property.textureScaleAndOffset;

            bool changed = false;
            float[] values = s_TempFloats[2];

            s_TempLabel.text = "Tiling";
            Rect vectorRect = EditorGUI.PrefixLabel(rect, s_TempLabel);
            values[0] = vector.x;
            values[1] = vector.y;
            EditorGUI.BeginChangeCheck();
            EditorGUI.MultiFloatField(vectorRect, s_XywhVectorLabels, values);
            if (EndProperty())
            {
                vector.x = values[0];
                vector.y = values[1];
                changed = true;
            }

            rect.y += rect.height + 2f;
            s_TempLabel.text = "Offset";
            vectorRect = EditorGUI.PrefixLabel(rect, s_TempLabel);
            values[0] = vector.z;
            values[1] = vector.w;
            EditorGUI.BeginChangeCheck();
            EditorGUI.MultiFloatField(vectorRect, s_XywhVectorLabels, values);
            if (EndProperty())
            {
                vector.z = values[0];
                vector.w = values[1];
                changed = true;
            }

            if (changed)
            {
                property.textureScaleAndOffset = vector;
            }

            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUI.indentLevel = indentLevel;
        }

        protected void DoUVSpeed(Rect rect, string[] names)
        {
            float labelWidth = EditorGUIUtility.labelWidth;
            int indentLevel = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            EditorGUIUtility.labelWidth = Mathf.Min(40f, rect.width * 0.20f);

            s_TempLabel.text = "Speed";
            rect = EditorGUI.PrefixLabel(rect, s_TempLabel);

            EditorGUIUtility.labelWidth = 13f;
            rect.width = rect.width * 0.5f - 1f;
            DoFloat(rect, names[0], "X");
            rect.x += rect.width + 2f;
            DoFloat(rect, names[1], "Y");
            EditorGUIUtility.labelWidth = labelWidth;
            EditorGUI.indentLevel = indentLevel;
        }

        protected void DoToggle(string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            s_TempLabel.text = label;
            bool value = EditorGUILayout.Toggle(s_TempLabel, property.floatValue == 1f);
            if (EndProperty())
            {
                property.floatValue = value ? 1f : 0f;
            }
        }

        protected void DoFloat(string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            Rect rect = EditorGUILayout.GetControlRect();
            rect.width = EditorGUIUtility.labelWidth + 55f;
            s_TempLabel.text = label;
            float value = EditorGUI.FloatField(rect, s_TempLabel, property.floatValue);
            if (EndProperty())
            {
                property.floatValue = value;
            }
        }

        protected void DoColor(string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            s_TempLabel.text = label;
            Color value = EditorGUI.ColorField(EditorGUILayout.GetControlRect(), s_TempLabel, property.colorValue);
            if (EndProperty())
            {
                property.colorValue = value;
            }
        }

        void DoFloat(Rect rect, string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            s_TempLabel.text = label;
            float value = EditorGUI.FloatField(rect, s_TempLabel, property.floatValue);
            if (EndProperty())
            {
                property.floatValue = value;
            }
        }

        protected void DoSlider(string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            Vector2 range = property.rangeLimits;
            s_TempLabel.text = label;
            float value = EditorGUI.Slider(
                EditorGUILayout.GetControlRect(), s_TempLabel, property.floatValue, range.x, range.y
            );
            if (EndProperty())
            {
                property.floatValue = value;
            }
        }

        protected void DoVector3(string name, string label)
        {
            MaterialProperty property = BeginProperty(name);
            s_TempLabel.text = label;
            Vector4 value = EditorGUILayout.Vector3Field(s_TempLabel, property.vectorValue);
            if (EndProperty())
            {
                property.vectorValue = value;
            }
        }

        protected void DoVector(string name, string label, GUIContent[] subLabels)
        {
            MaterialProperty property = BeginProperty(name);
            Rect rect = EditorGUILayout.GetControlRect();
            s_TempLabel.text = label;
            rect = EditorGUI.PrefixLabel(rect, s_TempLabel);
            Vector4 vector = property.vectorValue;

            float[] values = s_TempFloats[subLabels.Length];
            for (int i = 0; i < subLabels.Length; i++)
            {
                values[i] = vector[i];
            }

            EditorGUI.MultiFloatField(rect, subLabels, values);
            if (EndProperty())
            {
                for (int i = 0; i < subLabels.Length; i++)
                {
                    vector[i] = values[i];
                }

                property.vectorValue = vector;
            }
        }

        void DoDragAndDropBegin()
        {
            m_DragAndDropMinY = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true)).y;
        }

        void DoDragAndDropEnd()
        {
            Rect rect = GUILayoutUtility.GetRect(0f, 0f, GUILayout.ExpandWidth(true));
            Event evt = Event.current;
            if (evt.type == EventType.DragUpdated)
            {
                DragAndDrop.visualMode = DragAndDropVisualMode.Generic;
                evt.Use();
            }
            else if (
                evt.type == EventType.DragPerform &&
                Rect.MinMaxRect(rect.xMin, m_DragAndDropMinY, rect.xMax, rect.yMax).Contains(evt.mousePosition)
            )
            {
                DragAndDrop.AcceptDrag();
                evt.Use();
                Material droppedMaterial = DragAndDrop.objectReferences[0] as Material;
                if (droppedMaterial && droppedMaterial != m_Material)
                {
                    PerformDrop(droppedMaterial);
                }
            }
        }

        void PerformDrop(Material droppedMaterial)
        {
            Texture droppedTex = droppedMaterial.GetTexture(ShaderUtilities.ID_MainTex);
            if (!droppedTex)
            {
                return;
            }

            Texture currentTex = m_Material.GetTexture(ShaderUtilities.ID_MainTex);
            TMP_FontAsset requiredFontAsset = null;
            if (droppedTex != currentTex)
            {
                requiredFontAsset = TMP_EditorUtility.FindMatchingFontAsset(droppedMaterial);
                if (!requiredFontAsset)
                {
                    return;
                }
            }

            foreach (GameObject o in Selection.gameObjects)
            {
                if (requiredFontAsset)
                {
                    TMP_Text textComponent = o.GetComponent<TMP_Text>();
                    if (textComponent)
                    {
                        Undo.RecordObject(textComponent, "Font Asset Change");
                        textComponent.font = requiredFontAsset;
                    }
                }

                TMPro_EventManager.ON_DRAG_AND_DROP_MATERIAL_CHANGED(o, m_Material, droppedMaterial);
                EditorUtility.SetDirty(o);
            }
        }
    }
}
