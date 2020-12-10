using UnityEngine;
using UnityEngine.Events;

#if UNITY_EDITOR
using UnityEditor;
using Cinemachine.Editor;
#endif

namespace Cinemachine.Examples
{
    public class InvokeEventFromInspector : MonoBehaviour
    {
        public UnityEvent Event = new UnityEvent();
        public void Invoke() { Event.Invoke(); }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(InvokeEventFromInspector))]
    public class GenerateEventEditor : BaseEditor<InvokeEventFromInspector>
    {
        public override void OnInspectorGUI()
        {
            BeginInspector();
            Rect rect = EditorGUILayout.GetControlRect(true);
            if (GUI.Button(rect, "Invoke", "Button"))
                Target.Invoke();
            DrawRemainingPropertiesInInspector();
        }
    }
#endif
}
