#if MLA_INPUT_SYSTEM
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.Utilities;
using UnityEngine.Scripting;
#if UNITY_EDITOR
using UnityEditor;
#endif
namespace Unity.MLAgents.Extensions.Runtime.Input.Composites
{
    /// <summary>
    /// Custom composite class for reading values from <see cref="InputControl{Vector2}"/>s.
    /// </summary>
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    [Preserve]
    [DisplayStringFormat("{Vector2}")]
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
        public static void Init() { }

        public override Vector2 ReadValue(ref InputBindingCompositeContext context)
        {
            return context.ReadValue<Vector2, Vector2MagnitudeComparer>(Vector2);
        }
    }
}
#endif // MLA_INPUT_SYSTEM
