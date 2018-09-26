using UnityEngine;
using UnityEditor;
using System;
using System.Linq;
using UnityEditor.SceneManagement;

namespace MLAgents
{
    [CustomPropertyDrawer(typeof(BroadcastHub))]
    public class BroadcastHubDrawer : PropertyDrawer
    {
        private BroadcastHub hub;
        private const float lineHeight = 17f;

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            var addOne = (hub.Count > 0) ? 1 : 0;
            return (hub.Count + 2 + addOne) * lineHeight + 10f;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            CheckInitialize(property, label);
            position.height = lineHeight;
            EditorGUI.LabelField(position, new GUIContent(label.text, 
                "The Broadcast Hub helps you define which Brains you want to expose to " +
                "the external process"));

            EditorGUI.BeginProperty(position, label, property);
            // This is the rectangle for the Add button
            position.y += lineHeight;
            var addButtonRect = position;
            addButtonRect.x += 20;
            if (hub.Count > 0)
            {
                addButtonRect.width /= 2;
                addButtonRect.width -= 24;
                if (GUI.Button(addButtonRect, new GUIContent("Add New",
                    "Add a new Brain to the Broadcast Hub"), EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddNewItem();
                }

                // This is the rectangle for the Remove button
                var removeButtonRect = position;
                removeButtonRect.x = position.width / 2 + 15;
                removeButtonRect.width = addButtonRect.width - 18;
                if (GUI.Button(removeButtonRect, new GUIContent("Remove Last",
                        "Remove the last Brain from the Broadcast Hub"),
                    EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    RemoveLastItem();
                }
            }
            else
            {
                addButtonRect.width -= 50;
                if (GUI.Button(addButtonRect, new GUIContent("Add Brain to Broadcast Hub",
                    "Add a new Brain to the Broadcast Hub"), EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddNewItem();
                }
            }
            
            // This is the labels for each columns
            position.y += lineHeight;
            var labelRect = position;
            if (hub.Count > 0)
            {
                labelRect.x += 40;
                labelRect.width -= 144;
                EditorGUI.LabelField(labelRect, "Brains");
                labelRect = position;
                labelRect.x = position.width - 84;
                labelRect.width = 80;
                EditorGUI.LabelField(labelRect, "Control");
            }

            
            // Iterate over the elements
            for (var index = 0; index < hub.Count; index++)
            {
                var exposedBrains = hub.broadcastingBrains;
                var brain = exposedBrains[index];
                position.y += lineHeight;

                // This is the rectangle for the key
                var keyRect = position;
                keyRect.x += 20;
                keyRect.width -= 144;
                EditorGUI.BeginChangeCheck();
                var newBrain = EditorGUI.ObjectField(
                    keyRect, brain, typeof(Brain), true) as Brain;
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                    try
                    {
                        hub.broadcastingBrains.RemoveAt(index);
                        if (!exposedBrains.Contains(newBrain))
                        {
                            exposedBrains.Insert(index, newBrain);
                        }
                        else
                        {
                            exposedBrains.Insert(index, null);
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
                valueRect.x = position.width - 64;
                valueRect.width = 80;
                EditorGUI.BeginChangeCheck();
                if (brain != null)
                {
                    if (brain is LearningBrain)
                    {
                        var isTraining = hub.IsControlled(brain);
                        isTraining = 
                            EditorGUI.Toggle(valueRect, isTraining);
                        hub.SetTraining(brain, isTraining);
                    }
                }

                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                }

            }

            EditorGUI.EndProperty();
        }

        private void CheckInitialize(SerializedProperty property, GUIContent label)
        {
            if (hub == null)
            {
                var target = property.serializedObject.targetObject;
                hub = fieldInfo.GetValue(target) as BroadcastHub;
                if (hub == null)
                {
                    hub = new BroadcastHub();
                    fieldInfo.SetValue(target, hub);
                }
            }
        }
        
        private static void MarkSceneAsDirty()
        {
            if (!EditorApplication.isPlaying)
            {
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            }
        }

        private void ClearAllBrains()
        {
            hub.Clear();
        }

        private void RemoveLastItem()
        {
            if (hub.Count > 0)
            {
                hub.broadcastingBrains.RemoveAt(hub.broadcastingBrains.Count - 1);
            }
        }

        private void AddNewItem()
        {
            try
            {
                hub.broadcastingBrains.Add(null);
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }
        }
    }
}