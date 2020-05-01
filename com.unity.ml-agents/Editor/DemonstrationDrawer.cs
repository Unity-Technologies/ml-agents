using System.Collections.Generic;
using System.Text;
using UnityEditor;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;


namespace Unity.MLAgents.Editor
{
    /// <summary>
    /// Renders a custom UI for DemonstrationSummary ScriptableObject.
    /// </summary>
    [CustomEditor(typeof(DemonstrationSummary))]
    [CanEditMultipleObjects]
    internal class DemonstrationEditor : UnityEditor.Editor
    {
        SerializedProperty m_BrainParameters;
        SerializedProperty m_DemoMetaData;
        SerializedProperty m_ObservationShapes;

        void OnEnable()
        {
            m_BrainParameters = serializedObject.FindProperty("brainParameters");
            m_DemoMetaData = serializedObject.FindProperty("metaData");
            m_ObservationShapes = serializedObject.FindProperty("observationSummaries");
        }

        /// <summary>
        /// Renders Inspector UI for Demonstration metadata.
        /// </summary>
        void MakeMetaDataProperty(SerializedProperty property)
        {
            var nameProp = property.FindPropertyRelative("demonstrationName");
            var experiencesProp = property.FindPropertyRelative("numberSteps");
            var episodesProp = property.FindPropertyRelative("numberEpisodes");
            var rewardsProp = property.FindPropertyRelative("meanReward");

            var nameLabel = nameProp.displayName + ": " + nameProp.stringValue;
            var experiencesLabel = experiencesProp.displayName + ": " + experiencesProp.intValue;
            var episodesLabel = episodesProp.displayName + ": " + episodesProp.intValue;
            var rewardsLabel = rewardsProp.displayName + ": " + rewardsProp.floatValue;

            EditorGUILayout.LabelField(nameLabel);
            EditorGUILayout.LabelField(experiencesLabel);
            EditorGUILayout.LabelField(episodesLabel);
            EditorGUILayout.LabelField(rewardsLabel);
        }

        /// <summary>
        /// Constructs label for a serialized integer array.
        /// </summary>
        static string BuildIntArrayLabel(SerializedProperty actionSizeProperty)
        {
            var actionSize = actionSizeProperty.arraySize;
            var actionLabel = new StringBuilder("[ ");
            for (var i = 0; i < actionSize; i++)
            {
                actionLabel.Append(actionSizeProperty.GetArrayElementAtIndex(i).intValue);
                if (i < actionSize - 1)
                {
                    actionLabel.Append(", ");
                }
            }

            actionLabel.Append(" ]");
            return actionLabel.ToString();
        }

        /// <summary>
        /// Renders Inspector UI for BrainParameters of a DemonstrationSummary.
        /// Only the Action size and type are used from the BrainParameters.
        /// </summary>
        void MakeActionsProperty(SerializedProperty property)
        {
            var actSizeProperty = property.FindPropertyRelative("VectorActionSize");
            var actSpaceTypeProp = property.FindPropertyRelative("VectorActionSpaceType");

            var vecActSizeLabel =
                actSizeProperty.displayName + ": " + BuildIntArrayLabel(actSizeProperty);
            var actSpaceTypeLabel = actSpaceTypeProp.displayName + ": " +
                (SpaceType)actSpaceTypeProp.enumValueIndex;

            EditorGUILayout.LabelField(vecActSizeLabel);
            EditorGUILayout.LabelField(actSpaceTypeLabel);
        }

        /// <summary>
        /// Render the observation shapes of a DemonstrationSummary.
        /// </summary>
        /// <param name="obsSummariesProperty"></param>
        void MakeObservationsProperty(SerializedProperty obsSummariesProperty)
        {
            var shapesLabels = new List<string>();
            var numObservations = obsSummariesProperty.arraySize;
            for (var i = 0; i < numObservations; i++)
            {
                var summary = obsSummariesProperty.GetArrayElementAtIndex(i);
                var shapeProperty = summary.FindPropertyRelative("shape");
                shapesLabels.Add(BuildIntArrayLabel(shapeProperty));
            }

            var shapeLabel = $"Shapes: {string.Join(",  ", shapesLabels)}";
            EditorGUILayout.LabelField(shapeLabel);

        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.LabelField("Meta Data", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            MakeMetaDataProperty(m_DemoMetaData);
            EditorGUI.indentLevel--;

            EditorGUILayout.LabelField("Observations", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            MakeObservationsProperty(m_ObservationShapes);
            EditorGUI.indentLevel--;

            EditorGUILayout.LabelField("Actions", EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            MakeActionsProperty(m_BrainParameters);
            EditorGUI.indentLevel--;

            serializedObject.ApplyModifiedProperties();
        }
    }
}
