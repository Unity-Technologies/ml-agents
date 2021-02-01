using System.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.Utilities;
using UnityEngine.Scripting;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Extensions.Runtime.Input.Composites
{
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    [Preserve]
    [DisplayStringFormat("{Vector2}")] // This results in WASD.
    public class Vector2ValueComposite : InputBindingComposite<Vector2>
    {
        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once FieldCanBeMadeReadOnly.Global
        [InputControl(offset = 0, layout = "Vector2")] public int Vector2;

        static Vector2ValueComposite()
        {
            InputSystem.RegisterBindingComposite<Vector2ValueComposite>("Vector2Value");
        }

        [RuntimeInitializeOnLoadMethod]
        static void Init() { }

        public override Vector2 ReadValue(ref InputBindingCompositeContext context)
        {
            return context.ReadValue<Vector2, Vector2MagnitudeComparer>(Vector2);
        }
    }
}
