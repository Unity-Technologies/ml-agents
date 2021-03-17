using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
public class CustomEventSystem : EventSystem
{
    protected override void Awake()
    {
        base.Awake();
        unsafe
        {
            InputSystem.onDeviceCommand += InputSystemOnDeviceCommand;
        }
    }

    static unsafe long? InputSystemOnDeviceCommand(InputDevice device, InputDeviceCommand* command)
    {
        if (command->type != QueryCanRunInBackground.Type)
        {
            // return null is skip this evaluation
            return null;
        }
        ((QueryCanRunInBackground*)command)->canRunInBackground = true;
        return InputDeviceCommand.GenericSuccess;
    }
    protected override void OnApplicationFocus(bool hasFocus)
    {
        //Do not change focus flag on eventsystem
    }
}
