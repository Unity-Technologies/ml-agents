// From https://forum.unity.com/threads/finally-a-serializable-dictionary-for-unity-extracted-from-system-collections-generic.335797/
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;





[CustomPropertyDrawer(typeof(ResetParameters))]
public class ResetParameterDrawer : PropertyDrawer
{
    private ResetParameters _Dictionary;
    private bool _Foldout;
    private const float kButtonWidth = 18f;

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        CheckInitialize(property, label);
        if (_Foldout)
            return (_Dictionary.Count + 1) * 17f;
        return 17f;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        CheckInitialize(property, label);

        position.height = 17f;

        var foldoutRect = position;
        foldoutRect.width -= 2 * kButtonWidth;
        EditorGUI.BeginChangeCheck();
        _Foldout = EditorGUI.Foldout(foldoutRect, _Foldout, label, true);
        if (EditorGUI.EndChangeCheck())
            EditorPrefs.SetBool(label.text, _Foldout);

        var buttonRect = position;
        buttonRect.x = position.width - kButtonWidth + position.x;
        buttonRect.width = kButtonWidth + 2;

        if (GUI.Button(buttonRect, new GUIContent("+", "Add item"), EditorStyles.miniButton))
        {
            AddNewItem();
        }

        buttonRect.x -= kButtonWidth;

        if (GUI.Button(buttonRect, new GUIContent("X", "Clear dictionary"), EditorStyles.miniButtonRight))
        {
            ClearDictionary();
        }

        if (!_Foldout)
            return;

        foreach (var item in _Dictionary)
        {
            var key = item.Key;
            var value = item.Value;

            position.y += 17f;

            var keyRect = position;
            keyRect.width /= 2;
            keyRect.width -= 4;
            EditorGUI.BeginChangeCheck();
            var newKey = DoField(keyRect, typeof(string), key);
            if (EditorGUI.EndChangeCheck())
            {
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

            var valueRect = position;
            valueRect.x = position.width / 2 + 15;
            valueRect.width = keyRect.width - kButtonWidth;
            EditorGUI.BeginChangeCheck();
            value = DoField(valueRect, typeof(float), value);
            if (EditorGUI.EndChangeCheck())
            {
                _Dictionary[key] = value;
                break;
            }

            var removeRect = valueRect;
            removeRect.x = valueRect.xMax + 2;
            removeRect.width = kButtonWidth;
            if (GUI.Button(removeRect, new GUIContent("x", "Remove item"), EditorStyles.miniButtonRight))
            {
                RemoveItem(key);
                break;
            }
        }
    }

    private void RemoveItem(string key)
    {
        _Dictionary.Remove(key);
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

            _Foldout = EditorPrefs.GetBool(label.text);
        }
    }

    private static readonly Dictionary<Type, Func<Rect, object, object>> _Fields =
        new Dictionary<Type, Func<Rect, object, object>>()
        {
            { typeof(int), (rect, value) => EditorGUI.IntField(rect, (int)value) },
            { typeof(float), (rect, value) => EditorGUI.FloatField(rect, (float)value) },
            { typeof(string), (rect, value) => EditorGUI.TextField(rect, (string)value) },
            { typeof(bool), (rect, value) => EditorGUI.Toggle(rect, (bool)value) },
            { typeof(Vector2), (rect, value) => EditorGUI.Vector2Field(rect, GUIContent.none, (Vector2)value) },
            { typeof(Vector3), (rect, value) => EditorGUI.Vector3Field(rect, GUIContent.none, (Vector3)value) },
            { typeof(Bounds), (rect, value) => EditorGUI.BoundsField(rect, (Bounds)value) },
            { typeof(Rect), (rect, value) => EditorGUI.RectField(rect, (Rect)value) },
        };

    private static T DoField<T>(Rect rect, Type type, T value)
    {
        Func<Rect, object, object> field;
        if (_Fields.TryGetValue(type, out field))
            return (T)field(rect, value);

        if (type.IsEnum)
            return (T)(object)EditorGUI.EnumPopup(rect, (Enum)(object)value);

        if (typeof(UnityEngine.Object).IsAssignableFrom(type))
            return (T)(object)EditorGUI.ObjectField(rect, (UnityEngine.Object)(object)value, type, true);

        Debug.Log("Type is not supported: " + type);
        return value;
    }

    private void ClearDictionary()
    {
        _Dictionary.Clear();
    }

    private void AddNewItem()
    {
        string key = "Param";
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