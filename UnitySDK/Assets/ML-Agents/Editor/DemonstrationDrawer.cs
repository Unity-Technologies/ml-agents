using System.Text;
using MLAgents;
using UnityEditor;

/// <summary>
/// Renders a custom UI for Demonstration Scriptable Object.
/// </summary>
[CustomEditor(typeof(Demonstration))]
[CanEditMultipleObjects]
public class DemonstrationEditor : Editor
{
    SerializedProperty brainParameters;
    SerializedProperty demoMetaData;

    void OnEnable()
    {
        brainParameters = serializedObject.FindProperty("brainParameters");
        demoMetaData = serializedObject.FindProperty("metaData");
    }

    /// <summary>
    /// Renders Inspector UI for Demonstration metadata.
    /// </summary>
    void MakeMetaDataProperty(SerializedProperty property)
    {
        var nameProp = property.FindPropertyRelative("demonstrationName");
        var expProp = property.FindPropertyRelative("numberExperiences");
        var epiProp = property.FindPropertyRelative("numberEpisodes");
        var rewProp = property.FindPropertyRelative("meanReward");

        var nameLabel = nameProp.displayName + ": " + nameProp.stringValue;
        var expLabel = expProp.displayName + ": " + expProp.intValue;
        var epiLabel = epiProp.displayName + ": " + epiProp.intValue;
        var rewLabel = rewProp.displayName + ": " + rewProp.floatValue;

        EditorGUILayout.LabelField(nameLabel);
        EditorGUILayout.LabelField(expLabel);
        EditorGUILayout.LabelField(epiLabel);
        EditorGUILayout.LabelField(rewLabel);
    }

    /// <summary>
    /// Constructs label for action size array.
    /// </summary>
    static string BuildActionArrayLabel(SerializedProperty actionSizeProperty)
    {
        var actionSize = actionSizeProperty.arraySize;
        StringBuilder actionLabel = new StringBuilder("[ ");
        for (int i = 0; i < actionSize; i++)
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
    /// Renders Inspector UI for Brain Parameters of Demonstration.
    /// </summary>
    void MakeBrainParametersProperty(SerializedProperty property)
    {
        var vecObsSizeProp = property.FindPropertyRelative("vectorObservationSize");
        var numStackedProp = property.FindPropertyRelative("numStackedVectorObservations");
        var actSizeProperty = property.FindPropertyRelative("vectorActionSize");
        var camResProp = property.FindPropertyRelative("cameraResolutions");
        var actSpaceTypeProp = property.FindPropertyRelative("vectorActionSpaceType");

        var vecObsSizeLabel = vecObsSizeProp.displayName + ": " + vecObsSizeProp.intValue;
        var numStackedLabel = numStackedProp.displayName + ": " + numStackedProp.intValue;
        var vecActSizeLabel = actSizeProperty.displayName + ": " + BuildActionArrayLabel(actSizeProperty);
        var camResLabel = camResProp.displayName + ": " + camResProp.arraySize;
        var actSpaceTypeLabel = actSpaceTypeProp.displayName + ": " + (SpaceType) actSpaceTypeProp.enumValueIndex;

        EditorGUILayout.LabelField(vecObsSizeLabel);
        EditorGUILayout.LabelField(numStackedLabel);
        EditorGUILayout.LabelField(vecActSizeLabel);
        EditorGUILayout.LabelField(camResLabel);
        EditorGUILayout.LabelField(actSpaceTypeLabel);
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        EditorGUILayout.LabelField("Meta Data", EditorStyles.boldLabel);
        MakeMetaDataProperty(demoMetaData);
        EditorGUILayout.LabelField("Brain Parameters", EditorStyles.boldLabel);
        MakeBrainParametersProperty(brainParameters);
        serializedObject.ApplyModifiedProperties();
    }
}
