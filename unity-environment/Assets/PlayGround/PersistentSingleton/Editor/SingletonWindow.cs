using UnityEngine;
using UnityEditor;

class SingletonTestWindow : EditorWindow
{
    [MenuItem("Window/SingletonTest")]

    public static void ShowWindow()
    {
        EditorWindow.GetWindow(typeof(SingletonTestWindow));
    }

    void OnGUI()
    {
        SingletonTest.something = GUILayout.HorizontalSlider(SingletonTest.something, 1, 5);
        GUILayout.Label("Current value of something is " + SingletonTest.something);
    }
}