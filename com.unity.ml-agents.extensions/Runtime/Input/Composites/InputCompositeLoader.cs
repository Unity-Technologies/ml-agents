#if MLA_INPUT_SYSTEM
using System;

namespace Unity.MLAgents.Extensions.Runtime.Input.Composites
{
    public class InputCompositeLoader
    {
        static Lazy<InputCompositeLoader> s_Lazy = new Lazy<InputCompositeLoader>(() => new InputCompositeLoader());
        public static InputCompositeLoader Instance => s_Lazy.Value;

        static InputCompositeLoader()
        {
        }

        public void Init()
        {
            Vector2ValueComposite.Init();
        }

    }
}
#endif // MLA_INPUT_SYSTEM
