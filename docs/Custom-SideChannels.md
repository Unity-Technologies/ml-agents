# Custom SideChannels

You can create your own `SideChannel` in C# and Python and use it to communicate data between the two.

## Unity side
The side channel will have to implement the `SideChannel` abstract class and the following method.

 * `OnMessageReceived(byte[] data)` : You must implement this method to specify what the side channel will be doing
 with the data received from Python. The data is a `byte[]` argument.

The side channel must also assign a `ChannelId` property in the constructor. The `ChannelId` is a Guid
(or UUID in Python) used to uniquely identify a side channel. This Guid must be the same on C# and Python.
There can only be one side channel of a certain id during communication.

To send a byte array from C# to Python, call the `base.QueueMessageToSend(data)` method inside the side channel.
The `data` argument must be a `byte[]`.

To register a side channel on the Unity side, call `Academy.Instance.RegisterSideChannel` with the side channel
as only argument.

## Python side
The side channel will have to implement the `SideChannel` abstract class. You must implement :

 * `on_message_received(self, data: bytes) -> None` : You must implement this method to specify what the
 side channel will be doing with the data received from Unity. The data is a `byte[]` argument.

The side channel must also assign a `channel_id` property in the constructor. The `channel_id` is a UUID
(referred in C# as Guid) used to uniquely identify a side channel. This number must be the same on C# and
Python. There can only be one side channel of a certain id during communication.

To assign the `channel_id` call the abstract class constructor with the appropriate `channel_id` as follows:

```python
super().__init__(my_channel_id)
```

To send a byte array from Python to C#, call the `super().queue_message_to_send(bytes_data)` method inside the
side channel. The `bytes_data` argument must be a `bytes` object.

To register a side channel on the Python side, pass the side channel as argument when creating the
`UnityEnvironment` object. One of the arguments of the constructor (`side_channels`) is a list of side channels.

## Example implementation

Here is a simple implementation of a Side Channel that will exchange strings between C# and Python
(encoded as ascii).

One the C# side :
Here is an implementation of a `StringLogSideChannel` that will listed to the `UnityEngine.Debug.LogError` calls in
the game :

```csharp
using UnityEngine;
using MLAgents;
using System.Text;
using System;

public class StringLogSideChannel : SideChannel
{
    public StringLogSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    public override void OnMessageReceived(byte[] data)
    {
        var receivedString = Encoding.ASCII.GetString(data);
        Debug.Log("From Python : " + receivedString);
    }

    public void SendDebugStatementToPython(string logString, string stackTrace, LogType type)
    {
        if (type == LogType.Error)
        {
            var stringToSend = type.ToString() + ": " + logString + "\n" + stackTrace;
            var encodedString = Encoding.ASCII.GetBytes(stringToSend);
            base.QueueMessageToSend(encodedString);
        }
    }
}
```

We also need to register this side channel to the Academy and to the `Application.logMessageReceived` events,
so we write a simple MonoBehavior for this. (Do not forget to attach it to a GameObject in the scene).

```csharp
using UnityEngine;
using MLAgents;


public class RegisterStringLogSideChannel : MonoBehaviour
{

    StringLogSideChannel stringChannel;
    public void Awake()
    {
        // We create the Side Channel
        stringChannel = new StringLogSideChannel();

        // When a Debug.Log message is created, we send it to the stringChannel
        Application.logMessageReceived += stringChannel.SendDebugStatementToPython;

        // Just in case the Academy has not yet initialized
        Academy.Instance.RegisterSideChannel(stringChannel);
    }

    public void OnDestroy()
    {
        // De-register the Debug.Log callback
        Application.logMessageReceived -= stringChannel.SendDebugStatementToPython;
        if (Academy.IsInitialized){
            Academy.Instance.UnregisterSideChannel(stringChannel);
        }
    }

    public void Update()
    {
        // Optional : If the space bar is pressed, raise an error !
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.LogError("This is a fake error. Space bar was pressed in Unity.");
        }
    }
}
```

And here is the script on the Python side. This script creates a new Side channel type (`StringLogChannel`) and
launches a `UnityEnvironment` with that side channel.

```python

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel
import numpy as np
import uuid


# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, data: bytes) -> None:
        """
        Note :We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply print the data received interpreted as ascii
        print(data.decode("ascii"))

    def send_string(self, data: str) -> None:
        # Convert the string to ascii
        bytes_data = data.encode("ascii")
        # We call this method to queue the data we want to send
        super().queue_message_to_send(bytes_data)

# Create the channel
string_log = StringLogChannel()

# We start the communication with the Unity Editor and pass the string_log side channel as input
env = UnityEnvironment(base_port=UnityEnvironment.DEFAULT_EDITOR_PORT, side_channels=[string_log])
env.reset()
string_log.send_string("The environment was reset")

group_name = env.get_agent_groups()[0]  # Get the first group_name
for i in range(1000):
    step_data = env.get_step_result(group_name)
    n_agents = step_data.n_agents()  # Get the number of agents
    # We send data to Unity : A string with the number of Agent at each
    string_log.send_string(
        "Step " + str(i) + " occurred with " + str(n_agents) + " agents."
    )
    env.step()  # Move the simulation forward

env.close()
```

Now, if you run this script and press `Play` the Unity Editor when prompted, The console in the Unity Editor will
display a message at every Python step. Additionally, if you press the Space Bar in the Unity Engine, a message will
appear in the terminal.
