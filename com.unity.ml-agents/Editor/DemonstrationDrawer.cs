using System.Collections.Generic;
using System.Text;
using UnityEditor;
using Unity.MLAgents.Demonstrations;


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
        const string k_BrainParametersName = "brainParameters";
        const string k_MetaDataName = "metaData";
        const string k_ObservationSummariesName = "observationSummaries";
        const string k_DemonstrationName = "demonstrationName";
        const string k_NumberStepsName = "numberSteps";
        const string k_NumberEpisodesName = "numberEpisodes";
        const string k_MeanRewardName = "meanReward";
        const string k_ActionSpecName = "m_ActionSpec";
        const string k_NumContinuousActionsName = "m_NumContinuousActions";
        const string k_NumDiscreteActionsName = "BranchSizes";
        const string k_ShapeName = "shape";


        void OnEnable()
        {
            m_BrainParameters = serializedObject.FindProperty(k_BrainParametersName);
            m_DemoMetaData = serializedObject.FindProperty(k_MetaDataName);
            m_ObservationShapes = serializedObject.FindProperty(k_ObservationSummariesName);
        }

        /// <summary>
        /// Renders Inspector UI for Demonstration metadata.
        /// </summary>
        void MakeMetaDataProperty(SerializedProperty property)
        {
            var nameProp = property.FindPropertyRelative(k_DemonstrationName);
            var experiencesProp = property.FindPropertyRelative(k_NumberStepsName);
            var episodesProp = property.FindPropertyRelative(k_NumberEpisodesName);
            var rewardsProp = property.FindPropertyRelative(k_MeanRewardName);

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
            var actSpecProperty = property.FindPropertyRelative(k_ActionSpecName);
            var continuousSizeProperty = actSpecProperty.FindPropertyRelative(k_NumContinuousActionsName);
            var discreteSizeProperty = actSpecProperty.FindPropertyRelative(k_NumDiscreteActionsName);
            var continuousSizeLabel = "Continuous Actions: " + continuousSizeProperty.intValue;
            var discreteSizeLabel = "Discrete Action Branches: ";
            discreteSizeLabel += discreteSizeProperty == null ? "[]" : BuildIntArrayLabel(discreteSizeProperty);
            EditorGUILayout.LabelField(continuousSizeLabel);
            EditorGUILayout.LabelField(discreteSizeLabel);
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
                var shapeProperty = summary.FindPropertyRelative(k_ShapeName);
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
