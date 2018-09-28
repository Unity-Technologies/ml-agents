using UnityEngine;
using UnityEditor;
using System;
using System.Linq;
using UnityEditor.SceneManagement;

namespace MLAgents
{
    /// <summary>
    /// PropertyDrawer for ResetParameters. Defines how ResetParameters are displayed in the
    /// Inspector.
    /// </summary>
    [CustomPropertyDrawer(typeof(ResetParameters))]
    public class ResetParameterDrawer : PropertyDrawer
    {
        private ResetParameters _dict;
        private const float LineHeight = 17f;
        private const string NewKeyPrefix = "Param-";

        /// <summary>
        /// Computes the height of the Drawer depending on the property it is showing
        /// </summary>
        /// <param name="property">The property that is being drawn.</param>
        /// <param name="label">The label of the property being drawn.</param>
        /// <returns>The vertical space needed to draw the property.</returns>
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            return (_dict.Count + 2) * LineHeight;
        }

        /// <summary>
        /// Draws the ResetParameters property
        /// </summary>
        /// <param name="position">Rectangle on the screen to use for the property GUI.</param>
        /// <param name="property">The SerializedProperty of the ResetParameters
        /// to make the custom GUI for.</param>
        /// <param name="label">The label of this property.</param>
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            position.height = LineHeight;
            EditorGUI.LabelField(position, label);
            position.y += LineHeight;
            var width = position.width / 2 - 24;
            var keyRect = new Rect(position.x + 20, position.y, width, position.height);
            var valueRect = new Rect(position.x + width + 30, position.y, width, position.height);
            DrawButtons(keyRect, valueRect);
            EditorGUI.BeginProperty(position, label, property);
            foreach (var item in _dict)
            {
                var key = item.Key;
                var value = item.Value;
                keyRect.y += LineHeight;
                valueRect.y += LineHeight;
                EditorGUI.BeginChangeCheck();
                var newKey = EditorGUI.TextField(keyRect, key);
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                    try
                    {
                        _dict.Remove(key);
                        _dict.Add(newKey, value);
                    }
                    catch (Exception e)
                    {
                        Debug.Log(e.Message);
                    }
                    break;
                }

                EditorGUI.BeginChangeCheck();
                value = EditorGUI.FloatField(valueRect, value);
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                    _dict[key] = value;
                    break;
                }
            }
            EditorGUI.EndProperty();
        }

        /// <summary>
        /// Draws the Add and Remove buttons.
        /// </summary>
        /// <param name="addRect">The rectangle for the Add New button.</param>
        /// <param name="removeRect">The rectangle for the Remove Last button.</param>
        private void DrawButtons(Rect addRect, Rect removeRect)
        {
            // This is the Add button
            if (_dict.Count == 0)
            {
                addRect.width *= 2;
            }
            if (GUI.Button(addRect, new GUIContent("Add New",
                "Add a new item to the default reset parameters"), EditorStyles.miniButton))
            {
                MarkSceneAsDirty();
                AddNewItem();
            }
            
            // This is the Remove button
            if (_dict.Count == 0)
            {
                return;
            }
            if (GUI.Button(removeRect, new GUIContent("Remove Last",
                "Remove the last item from the default reset parameters"), EditorStyles.miniButton))
            {
                MarkSceneAsDirty();
                RemoveLastItem();
            }
        }

        /// <summary>
        /// Signals that the property has been modified and requires the scene to be saved for
        /// the changes to persist. Only works when the Editor is not playing.
        /// </summary>
        private static void MarkSceneAsDirty()
        {
            if (!EditorApplication.isPlaying)
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            }
        }

        /// <summary>
        /// Ensures that the state of the Drawer is synchronized with the property.
        /// </summary>
        /// <param name="property">The SerializedProperty of the ResetParameters
        /// to make the custom GUI for.</param>
        /// <param name="label">The label of this property.</param>
        private void CheckInitialize(SerializedProperty property, GUIContent label)
        {
            if (_dict == null)
            {
                var target = property.serializedObject.targetObject;
                _dict = fieldInfo.GetValue(target) as ResetParameters;
                if (_dict == null)
                {
                    _dict = new ResetParameters();
                    fieldInfo.SetValue(target, _dict);
                }
            }
        }

        /// <summary>
        /// Removes the last ResetParameter from the ResetParameters
        /// </summary>
        private void RemoveLastItem()
        {
            if (_dict.Count > 0)
            {
                string key = _dict.Keys.ToList()[_dict.Count - 1];
                _dict.Remove(key);
            }
        }

        /// <summary>
        /// Adds a new ResetParameter to the ResetParameters with a default name.
        /// </summary>
        private void AddNewItem()
        {
            string key = NewKeyPrefix + _dict.Count;
            var value = default(float);
            try
            {
                _dict.Add(key, value);
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
    }
}
