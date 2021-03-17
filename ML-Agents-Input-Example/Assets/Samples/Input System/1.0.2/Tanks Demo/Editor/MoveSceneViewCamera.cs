#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

public class MoveSceneViewCamera
{
    [MenuItem("Window/Position Scene View Camera")]
    static void PositionCamera()
    {
        SceneView.lastActiveSceneView.pivot = new Vector3(-147f, 23.5f, 237f);
        SceneView.lastActiveSceneView.rotation = Quaternion.Euler(0f, 150f, 0f);
        SceneView.lastActiveSceneView.orthographic = true;
        SceneView.lastActiveSceneView.size = 100f;
        Selection.activeGameObject = Camera.main.gameObject;
    }
}
#endif
