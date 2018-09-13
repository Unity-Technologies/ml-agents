using UnityEngine;
using UnityEditor;
using System;
using UnityEditor.SceneManagement;

namespace MLAgents
{
    [CustomPropertyDrawer(typeof(TrainingHub))]
    public class TrainingHubDrawer : PropertyDrawer
    {
        private TrainingHub hub;
        private const float lineHeight = 17f;
//        private const float keyValueRatio = 0.8f;

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            return (hub.brainsToTrain.Count + 3) * lineHeight;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {

            
            CheckInitialize(property, label);
            position.height = lineHeight;
            EditorGUI.LabelField(position, label);

            EditorGUI.BeginProperty(position, label, property);
            
            
            position.y += lineHeight;
            var labelRect = position;
            labelRect.x += 40;
            labelRect.width -= 104;
            EditorGUI.LabelField(labelRect, "Brains");
            labelRect = position;
            labelRect.x = position.width - 54;
            labelRect.width = 40;
            EditorGUI.LabelField(labelRect, "Train");

            
            for (var index = 0; index < hub.brainsToTrain.Count; index+=1)
            {
                var item = hub.brainsToTrain[index];
                position.y += lineHeight;

                // This is the rectangle for the key
                var keyRect = position;
                keyRect.x += 20;
                keyRect.width -= 104;
                EditorGUI.BeginChangeCheck();
                var newBrain = EditorGUI.ObjectField(
                    keyRect, item, typeof(Brain), true) as Brain;
                if (EditorGUI.EndChangeCheck())
                {
                    EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                    try
                    {
                        hub.brainsToTrain.RemoveAt(index);
                        if (!hub.brainsToTrain.Contains(newBrain))
                        {
                            hub.brainsToTrain.Insert(index, newBrain);
                        }
                        else
                        {
                            hub.brainsToTrain.Insert(index, null);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.Log(e.Message);
                    }

                    break;
                }

                // This is the Rectangle for the value
                var valueRect = position;
                valueRect.x = position.width - 44;
                valueRect.width = 40;
                EditorGUI.BeginChangeCheck();
                if (item != null)
                {
                    item.isExternal = EditorGUI.Toggle(valueRect, item.isExternal);
                }
                else
                {
                    EditorGUI.Toggle(valueRect, false);
                }

//                if (EditorGUI.EndChangeCheck())
//                {
//                    EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
//                    for (var i = 0; i < hub.brainsToTrain.Count; i++)
//                    {
//                        if (hub.brainsToTrain[i].brain == brain)
//                        {
//                            hub.brainsToTrain[i].train = external;
//                        }
//                    }
//
//                    hub.brainsToTrain
////                    _Dictionary[key] = value;
//                    break;
//                }
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
            if (hub == null)
            {
                var target = property.serializedObject.targetObject;
                hub = fieldInfo.GetValue(target) as TrainingHub;
                if (hub == null)
                {
                    hub = new TrainingHub();
                    fieldInfo.SetValue(target, hub);
                }
            }
        }

        private void ClearResetParamters()
        {
            hub.brainsToTrain.Clear();
        }

        private void RemoveLastItem()
        {
            if (hub.brainsToTrain.Count > 0)
            {
                hub.brainsToTrain.RemoveAt(hub.brainsToTrain.Count - 1);
            }
        }

        private void AddNewItem()
        {
            try
            {
                hub.brainsToTrain.Add(null);
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
    }
}