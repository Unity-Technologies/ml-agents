using UnityEngine;
using UnityEditor;

class StaticTestWindow : EditorWindow
{
    [MenuItem("Window/StaticTest")]

    public static void ShowWindow()
    {
        EditorWindow.GetWindow(typeof(StaticTestWindow));
    }

    void OnGUI()
    {
        StaticTest.something = GUILayout.HorizontalSlider(StaticTest.something, 1, 5);
        GUILayout.Label("Current value of something is " + StaticTest.something);
    }
}