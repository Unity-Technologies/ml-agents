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
        private ResetParameters _parameters;
        // The height of a line in the Unity Inspectors
        private const float LineHeight = 17f;
        // This is the prefix for the key when you add a reset parameter
        private const string NewKeyPrefix = "Param-";

        /// <summary>
        /// Computes the height of the Drawer depending on the property it is showing
        /// </summary>
        /// <param name="property">The property that is being drawn.</param>
        /// <param name="label">The label of the property being drawn.</param>
        /// <returns>The vertical space needed to draw the property.</returns>
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            LazyInitializeParameters(property, label);
            return (_parameters.Count + 2) * LineHeight;
        }

        /// <inheritdoc />
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            LazyInitializeParameters(property, label);
            position.height = LineHeight;
            EditorGUI.LabelField(position, label);
            position.y += LineHeight;
            var width = position.width / 2 - 24;
            var keyRect = new Rect(position.x + 20, position.y, width, position.height);
            var valueRect = new Rect(position.x + width + 30, position.y, width, position.height);
            DrawAddRemoveButtons(keyRect, valueRect);
            EditorGUI.BeginProperty(position, label, property);
            foreach (var parameter in _parameters)
            {
                var key = parameter.Key;
                var value = parameter.Value;
                keyRect.y += LineHeight;
                valueRect.y += LineHeight;
                EditorGUI.BeginChangeCheck();
                var newKey = EditorGUI.TextField(keyRect, key);
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                    try
                    {
                        _parameters.Remove(key);
                        _parameters.Add(newKey, value);
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
                    _parameters[key] = value;
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
        private void DrawAddRemoveButtons(Rect addRect, Rect removeRect)
        {
            // This is the Add button
            if (_parameters.Count == 0)
            {
                addRect.width *= 2;
            }
            if (GUI.Button(addRect,
                new GUIContent("Add New", "Add a new item to the default reset parameters"), 
                EditorStyles.miniButton))
            {
                MarkSceneAsDirty();
                AddParameter();
            }
            
            // If there are no items in the ResetParameters, Hide the Remove button
            if (_parameters.Count == 0)
            {
                return;
            }
            // This is the Remove button
            if (GUI.Button(removeRect, 
                new GUIContent(
                    "Remove Last", "Remove the last item from the default reset parameters"), 
                EditorStyles.miniButton))
            {
                MarkSceneAsDirty();
                RemoveLastParameter();
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
        private void LazyInitializeParameters(SerializedProperty property, GUIContent label)
        {
            if (_parameters != null)
            {
                return;
            }
            var target = property.serializedObject.targetObject;
            _parameters = fieldInfo.GetValue(target) as ResetParameters;
            if (_parameters == null)
            {
                _parameters = new ResetParameters();
                fieldInfo.SetValue(target, _parameters);
            }
        }

        /// <summary>
        /// Removes the last ResetParameter from the ResetParameters
        /// </summary>
        private void RemoveLastParameter()
        {
            if (_parameters.Count > 0)
            {
                string key = _parameters.Keys.ToList()[_parameters.Count - 1];
                _parameters.Remove(key);
            }
        }

        /// <summary>
        /// Adds a new ResetParameter to the ResetParameters with a default name.
        /// </summary>
        private void AddParameter()
        {
            string key = NewKeyPrefix + _parameters.Count;
            var value = default(float);
            try
            {
                _parameters.Add(key, value);
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
    }
}
