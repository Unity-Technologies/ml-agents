using UnityEngine.InputSystem;

namespace Unity.MLAgents.Extensions.Runtime.Input
{
    public interface IIntputActionAssetProvider
    {
        InputActionAsset GetInputActionAsset();
    }
}
