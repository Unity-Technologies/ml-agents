#if MLA_INPUT_SYSTEM
using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IIntputActionAssetProvider
    {
        InputActionAsset GetInputActionAsset();
    }
}
#endif // MLA_INPUT_SYSTEM
