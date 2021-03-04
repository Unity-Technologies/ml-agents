#if MLA_INPUT_SYSTEM && UNITY_2019_4_OR_NEWER
using UnityEngine.InputSystem.LowLevel;

namespace Unity.MLAgents.Extensions.Input
{
    public interface IInputEventPtrProvider
    {
        InputEventPtr GetEventForFrame();
        void EventWrittenToInFrame();
    }
}
#endif // MLA_INPUT_SYSTEM && UNITY_2019_4_OR_NEWER
