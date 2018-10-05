using UnityEngine;
using UnityEditor;
using System;
using System.Linq;
using UnityEditor.SceneManagement;

namespace MLAgents
{
    /// <summary>
    /// PropertyDrawer for BroadcastHub. Used to display the BroadcastHub in the Inspector.
    /// </summary>
    [CustomPropertyDrawer(typeof(BroadcastHub))]
    public class BroadcastHubDrawer : PropertyDrawer
    {
        private BroadcastHub _hub;
        // The height of a line in the Unity Inspectors
        private const float LineHeight = 17f;
        // The vertical space left below the BroadcastHub UI.
        private const float ExtraSpaceBelow = 10f;
        // The horizontal size of the Control checkbox
        private const int ControlSize = 80;

        /// <summary>
        /// Computes the height of the Drawer depending on the property it is showing
        /// </summary>
        /// <param name="property">The property that is being drawn.</param>
        /// <param name="label">The label of the property being drawn.</param>
        /// <returns>The vertical space needed to draw the property.</returns>
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            LazyInitializeHub(property, label);
            var numLines = _hub.Count + 2 + (_hub.Count > 0 ? 1 : 0);
            return (numLines) * LineHeight + ExtraSpaceBelow;
        }

        /// <inheritdoc />
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            LazyInitializeHub(property, label);
            position.height = LineHeight;
            EditorGUI.LabelField(position, new GUIContent(label.text, 
                "The Broadcast Hub helps you define which Brains you want to expose to " +
                "the external process"));
            position.y += LineHeight;

            EditorGUI.BeginProperty(position, label, property);

            EditorGUI.indentLevel++;
            DrawAddRemoveButtons(position);
            position.y += LineHeight;
            
            // This is the labels for each columns
            var brainWidth = position.width - ControlSize;
            var brainRect = new Rect(
                position.x, position.y, brainWidth, position.height);
            var controlRect = new Rect(
                position.x + brainWidth, position.y, ControlSize, position.height);
            if (_hub.Count > 0)
            {
                EditorGUI.LabelField(brainRect, "Brains");
                brainRect.y += LineHeight;
                EditorGUI.LabelField(controlRect, "Control");
                controlRect.y += LineHeight;
                controlRect.x += 15;
            }
            DrawBrains(brainRect, controlRect);
            EditorGUI.indentLevel--;
            EditorGUI.EndProperty();
        }
        
        /// <summary>
        /// Draws the Add and Remove buttons.
        /// </summary>
        /// <param name="position">The position at which to draw.</param>
        private void DrawAddRemoveButtons(Rect position)
        {
            // This is the rectangle for the Add button
            var addButtonRect = position;
            addButtonRect.x += 20;
            if (_hub.Count > 0)
            {
                addButtonRect.width /= 2;
                addButtonRect.width -= 24;
                var buttonContent = new GUIContent(
                    "Add New", "Add a new Brain to the Broadcast Hub");
                if (GUI.Button(addButtonRect, buttonContent, EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddBrain();
                }
                // This is the rectangle for the Remove button
                var removeButtonRect = position;
                removeButtonRect.x = position.width / 2 + 15;
                removeButtonRect.width = addButtonRect.width - 18;
                buttonContent = new GUIContent(
                    "Remove Last", "Remove the last Brain from the Broadcast Hub");
                if (GUI.Button(removeButtonRect, buttonContent, EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    RemoveLastBrain();
                }
            }
            else
            {
                addButtonRect.width -= 50;
                var buttonContent = new GUIContent(
                    "Add Brain to Broadcast Hub", "Add a new Brain to the Broadcast Hub");
                if (GUI.Button(addButtonRect, buttonContent, EditorStyles.miniButton))
                {
                    MarkSceneAsDirty();
                    AddBrain();
                }
            }
        }

        /// <summary>
        /// Draws the Brain and Control checkbox for the brains contained in the BroadCastHub.
        /// </summary>
        /// <param name="brainRect">The Rect to draw the Brains.</param>
        /// <param name="controlRect">The Rect to draw the control checkbox.</param>
        private void DrawBrains(Rect brainRect, Rect controlRect)
        {
            for (var index = 0; index < _hub.Count; index++)
            {
                var exposedBrains = _hub.broadcastingBrains;
                var brain = exposedBrains[index];
                // This is the rectangle for the brain
                EditorGUI.BeginChangeCheck();
                var newBrain = EditorGUI.ObjectField(
                    brainRect, brain, typeof(Brain), true) as Brain;
                brainRect.y += LineHeight;
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                    _hub.broadcastingBrains.RemoveAt(index);
                    var brainToInsert = exposedBrains.Contains(newBrain) ? null : newBrain;
                    exposedBrains.Insert(index, brainToInsert);
                    break;
                }
                // This is the Rectangle for the control checkbox
                EditorGUI.BeginChangeCheck();
                if (brain is LearningBrain)
                {
                    var isTraining = _hub.IsControlled(brain);
                    isTraining = EditorGUI.Toggle(controlRect, isTraining);
                    _hub.SetControlled(brain, isTraining);
                }
                controlRect.y += LineHeight;
                if (EditorGUI.EndChangeCheck())
                {
                    MarkSceneAsDirty();
                }
            }
        }

        /// <summary>
        /// Lazy initializes the Drawer with the property to be drawn.
        /// </summary>
        /// <param name="property">The SerializedProperty of the BroadcastHub
        /// to make the custom GUI for.</param>
        /// <param name="label">The label of this property.</param>
        private void LazyInitializeHub(SerializedProperty property, GUIContent label)
        {
            if (_hub != null)
            {
                return;
            }
            var target = property.serializedObject.targetObject;
            _hub = fieldInfo.GetValue(target) as BroadcastHub;
            if (_hub == null)
            {
                _hub = new BroadcastHub();
                fieldInfo.SetValue(target, _hub);
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
        /// Removes the last Brain from the BroadcastHub
        /// </summary>
        private void RemoveLastBrain()
        {
            if (_hub.Count > 0)
            {
                _hub.broadcastingBrains.RemoveAt(_hub.broadcastingBrains.Count - 1);
            }
        }

        /// <summary>
        /// Adds a new Brain to the BroadcastHub. The value of this brain will not be initialized.
        /// </summary>
        private void AddBrain()
        {
            _hub.broadcastingBrains.Add(null);
        }
    }
}
