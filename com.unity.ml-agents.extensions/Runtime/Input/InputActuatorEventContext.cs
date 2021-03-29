#if MLA_INPUT_SYSTEM
using System;
using Unity.Collections;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Extensions.Input
{
    /// <summary>
    /// This interface is passed to InputActionActuators to allow them to write to InputEvents.
    /// The way this interface should be used is to request the <see cref="InputEventPtr"/> by calling
    /// <see cref="GetEventForFrame"/> then call <see cref="EventProcessedInFrame"/> before returning from
    /// </summary>
    public class InputActuatorEventContext : IDisposable
    {

        /// <summary>
        /// The number of times to allow the use of an event before queuing it in the InputSystem.
        /// </summary>
        public readonly int NumTimesToProcess;
        public readonly InputDevice InputDevice;
        NativeArray<byte> m_EventBuffer;
        InputEventPtr m_Ptr;
        int m_Count;

#if UNITY_EDITOR
        public static InputActuatorEventContext s_EditorContext = new InputActuatorEventContext();
#endif

        public InputActuatorEventContext(int numTimesToProcess = 1, InputDevice device = null)
        {
            NumTimesToProcess = numTimesToProcess;
            InputDevice = device;
            m_Count = 0;
            m_Ptr = new InputEventPtr();
            m_EventBuffer = new NativeArray<byte>();
        }

        /// <summary>
        /// Returns the <see cref="InputEventPtr"/> to write to for the current frame.
        /// </summary>
        /// <returns>The <see cref="InputEventPtr"/> to write to for the current frame.</returns>
        public IDisposable GetEventForFrame(out InputEventPtr eventPtr)
        {
#if UNITY_EDITOR
            if (!EditorApplication.isPlaying)
            {
                eventPtr = new InputEventPtr();
            }
#endif
            if (m_Count % NumTimesToProcess == 0)
            {
                m_Count = 0;
                m_EventBuffer = StateEvent.From(InputDevice, out m_Ptr);
            }
            eventPtr = m_Ptr;
            return this;
        }

        public void Dispose()
        {
#if UNITY_EDITOR
            if (!EditorApplication.isPlaying)
            {
                return;
            }
#endif
            m_Count++;
            if (m_Count == NumTimesToProcess && m_Ptr.valid)
            {
                InputSystem.QueueEvent(m_Ptr);
                m_EventBuffer.Dispose();
            }

        }
    }
}
#endif // MLA_INPUT_SYSTEM
