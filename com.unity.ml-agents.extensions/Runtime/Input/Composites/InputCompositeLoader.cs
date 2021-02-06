#if MLA_INPUT_SYSTEM
using System;
using UnityEngine.Scripting;

namespace Unity.MLAgents.Extensions.Runtime.Input.Composites
{
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
