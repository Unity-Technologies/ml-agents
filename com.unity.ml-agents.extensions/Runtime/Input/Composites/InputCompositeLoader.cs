#if MLA_INPUT_SYSTEM
using System;
using UnityEngine.Scripting;

namespace Unity.MLAgents.Extensions.Runtime.Input.Composites
{
    /// <summary>
    /// Class that loads custom <see cref="UnityEngine.InputSystem.InputBindingComposite"/>s.
    /// </summary>
    [Preserve]
    public class InputCompositeLoader
    {
        public static void Init()
        {
            Vector2ValueComposite.Init();
            AxisValueComposite.Init();
        }

    }
}
#endif // MLA_INPUT_SYSTEM
