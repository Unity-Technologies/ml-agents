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
    /// Custom binding for reading float values from an
    /// </summary>
#if UNITY_EDITOR
    [InitializeOnLoad]
#endif
    [Preserve, DisplayStringFormat("{Axis}")]
    public class AxisValueComposite : InputBindingComposite<float>
    {
        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once FieldCanBeMadeReadOnly.Global
        [InputControl(offset = 0, layout = "Axis")] public int Axis;

        static AxisValueComposite()
        {
            InputSystem.RegisterBindingComposite<AxisValueComposite>("AxisValue");
        }

        [RuntimeInitializeOnLoadMethod]
        public static void Init() { }

        public override float ReadValue(ref InputBindingCompositeContext context)
        {
            return context.ReadValue<float>(Axis);
        }
    }
}
#endif // MLA_INPUT_SYSTEM
