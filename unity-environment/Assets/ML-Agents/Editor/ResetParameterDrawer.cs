using UnityEngine;
using UnityEditor;
using System;
using System.Linq;
using UnityEditor.SceneManagement;

namespace MLAgents
{

    [CustomPropertyDrawer(typeof(ResetParameters))]
    public class ResetParameterDrawer : PropertyDrawer
    {
        private ResetParameters _Dictionary;
        private const float lineHeight = 17f;

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            return (_Dictionary.Count + 2) * lineHeight;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {

            CheckInitialize(property, label);
            position.height = lineHeight;
            EditorGUI.LabelField(position, label);

            EditorGUI.BeginProperty(position, label, property);
            foreach (var item in _Dictionary)
            {
                var key = item.Key;
                var value = item.Value;
                position.y += lineHeight;

                // This is the rectangle for the key
                var keyRect = position;
                keyRect.x += 20;
                keyRect.width /= 2;
                keyRect.width -= 24;
                EditorGUI.BeginChangeCheck();
                var newKey = EditorGUI.TextField(keyRect, key);
                if (EditorGUI.EndChangeCheck())
                {
                    EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                    try
                    {
                        _Dictionary.Remove(key);
                        _Dictionary.Add(newKey, value);
                    }
                    catch (Exception e)
                    {
                        Debug.Log(e.Message);
                    }

                    break;
                }

                // This is the Rectangle for the value
                var valueRect = position;
                valueRect.x = position.width / 2 + 15;
                valueRect.width = keyRect.width - 18;
                EditorGUI.BeginChangeCheck();
                value = EditorGUI.FloatField(valueRect, value);
                if (EditorGUI.EndChangeCheck())
                {
                    EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                    _Dictionary[key] = value;
                    break;
                }
            }

            // This is the rectangle for the Add button
            position.y += lineHeight;
            var AddButtonRect = position;
            AddButtonRect.x += 20;
            AddButtonRect.width /= 2;
            AddButtonRect.width -= 24;
            if (GUI.Button(AddButtonRect, new GUIContent("Add New",
                "Add a new item to the default reset paramters"), EditorStyles.miniButton))
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                AddNewItem();
            }

            // This is the rectangle for the Remove button
            var RemoveButtonRect = position;
            RemoveButtonRect.x = position.width / 2 + 15;
            RemoveButtonRect.width = AddButtonRect.width - 18;
            if (GUI.Button(RemoveButtonRect, new GUIContent("Remove Last",
                "Remove the last item to the default reset paramters"), EditorStyles.miniButton))
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                RemoveLastItem();
            }


            EditorGUI.EndProperty();

        }

        private void CheckInitialize(SerializedProperty property, GUIContent label)
        {
            if (_Dictionary == null)
            {
                var target = property.serializedObject.targetObject;
                _Dictionary = fieldInfo.GetValue(target) as ResetParameters;
                if (_Dictionary == null)
                {
                    _Dictionary = new ResetParameters();
                    fieldInfo.SetValue(target, _Dictionary);
                }
            }
        }

        private void ClearResetParamters()
        {
            _Dictionary.Clear();
        }

        private void RemoveLastItem()
        {
            if (_Dictionary.Count > 0)
            {
                string key = _Dictionary.Keys.ToList()[_Dictionary.Count - 1];
                _Dictionary.Remove(key);
            }
        }

        private void AddNewItem()
        {
            string key = "Param-" + _Dictionary.Count.ToString();
            var value = default(float);
            try
            {
                _Dictionary.Add(key, value);
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
    }
}
