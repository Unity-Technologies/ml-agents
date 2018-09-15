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
            var addOne = (hub.brainsToTrain.Count > 0) ? 1 : 0;
            return (hub.brainsToTrain.Count + 2 + addOne) * lineHeight + 10f;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {

            
            CheckInitialize(property, label);
            position.height = lineHeight;
            EditorGUI.LabelField(position, label);

            EditorGUI.BeginProperty(position, label, property);
            // This is the rectangle for the Add button
            position.y += lineHeight;
            var addButtonRect = position;
            addButtonRect.x += 20;
            if (hub.brainsToTrain.Count > 0)
            {
                addButtonRect.width /= 2;
                addButtonRect.width -= 24;
                if (GUI.Button(addButtonRect, new GUIContent("Add New",
                    "Add a new item to the default reset paramters"), EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddNewItem();
                }

                // This is the rectangle for the Remove button
                var RemoveButtonRect = position;
                RemoveButtonRect.x = position.width / 2 + 15;
                RemoveButtonRect.width = addButtonRect.width - 18;
                if (GUI.Button(RemoveButtonRect, new GUIContent("Remove Last",
                        "Remove the last item to the default reset paramters"),
                    EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    RemoveLastItem();
                }
            }
            else
            {
                addButtonRect.width -= 50;
                if (GUI.Button(addButtonRect, new GUIContent("Add Brain to Training Session",
                    "Add a new item to the default reset paramters"), EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddNewItem();
                }
            }
            
            // This is the labels for each columns
            position.y += lineHeight;
            var labelRect = position;
            if (hub.brainsToTrain.Count > 0)
            {
                labelRect.x += 40;
                labelRect.width -= 104;
                EditorGUI.LabelField(labelRect, "Brains");
                labelRect = position;
                labelRect.x = position.width - 54;
                labelRect.width = 40;
                EditorGUI.LabelField(labelRect, "Train");
            }

            
            // Iterate over the elements
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
                    MarkSceneAsDirty();
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
                    if (item is InternalBrain)
                    {
                        item.isExternal = EditorGUI.Toggle(valueRect, item.isExternal);
                    }
                    else
                    {
                        item.isExternal = false;
                    }
                }

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
        
        private static void MarkSceneAsDirty()
        {
            try
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            }
            catch (Exception e)
            {
                
            }
        }

        private void ClearAllBrains()
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