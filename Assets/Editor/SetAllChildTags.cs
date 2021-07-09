using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Experimental.SceneManagement;

public class SetAllChildTags : ScriptableWizard
{
    public GameObject objToChange;
    public string tag;

    void OnWizardUpdate()
    {
        helpString = "Sets all child tags of a GameObject.";
    }

    void OnWizardCreate()
    {
        var x = objToChange.GetComponentsInChildren<Transform>();
        foreach (var z in x) z.tag = tag;
        // Make sure prefab is saved
        var prefabStage = PrefabStageUtility.GetCurrentPrefabStage();
        if (prefabStage != null)
        {
            EditorSceneManager.MarkSceneDirty(prefabStage.scene);
        }
    }

    [MenuItem("GameObject/Set All Child Tags")]
    static void TagRecursively()
    {
        ScriptableWizard.DisplayWizard("Change Tags Recursively", typeof(SetAllChildTags), "Apply");
    }
}
