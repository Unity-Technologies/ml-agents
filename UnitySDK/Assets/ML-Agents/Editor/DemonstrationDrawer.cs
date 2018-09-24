using MLAgents;
using UnityEditor;

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

    void MakeMetaDataProperty(SerializedProperty property)
    {
        var nameString = property.FindPropertyRelative("demonstrationName").displayName + ": " +
                         property.FindPropertyRelative("demonstrationName").stringValue;
        
        var expString = property.FindPropertyRelative("numberExperiences").displayName + ": " +
                        property.FindPropertyRelative("numberExperiences").intValue;
        
        var epiString = property.FindPropertyRelative("numberEpisodes").displayName + ": " +
                        property.FindPropertyRelative("numberEpisodes").intValue;
        
        var rewString = property.FindPropertyRelative("meanReward").displayName + ": " +
                        property.FindPropertyRelative("meanReward").floatValue;
        
        
        EditorGUILayout.LabelField(nameString);
        EditorGUILayout.LabelField(expString);
        EditorGUILayout.LabelField(epiString);
        EditorGUILayout.LabelField(rewString);
    }

    void MakeBrainParametersProperty(SerializedProperty property)
    {
        var vecObsSizeS = property.FindPropertyRelative("vectorObservationSize").displayName + ": " +
                         property.FindPropertyRelative("vectorObservationSize").intValue;
        
        var numStackedS = property.FindPropertyRelative("numStackedVectorObservations").displayName + ": " +
                          property.FindPropertyRelative("numStackedVectorObservations").intValue;
        
        var vecActSizeS = property.FindPropertyRelative("vectorActionSize").displayName + ": " +
                          property.FindPropertyRelative("vectorActionSize").arraySize;
        
        var camResS = property.FindPropertyRelative("cameraResolutions").displayName + ": " +
                          property.FindPropertyRelative("cameraResolutions").arraySize;
        
        var actSpaceTypeS = property.FindPropertyRelative("vectorActionSpaceType").displayName + ": " +
                      (SpaceType) property.FindPropertyRelative("vectorActionSpaceType").enumValueIndex;
        
        EditorGUILayout.LabelField(vecObsSizeS);
        EditorGUILayout.LabelField(numStackedS);
        EditorGUILayout.LabelField(vecActSizeS);
        EditorGUILayout.LabelField(camResS);
        EditorGUILayout.LabelField(actSpaceTypeS);
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
