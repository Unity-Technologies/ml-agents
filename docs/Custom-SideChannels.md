# Custom Side Channels

You can create your own side channel in C# and Python and use it to communicate
custom data structures between the two. This can be useful for situations in
which the data to be sent is too complex or structured for the built-in
`FloatPropertiesChannel`, or is not related to any specific agent, and therefore
inappropriate as an agent observation.

## Overview

In order to use a side channel, it must be implemented as both Unity and Python classes.

### Unity side
The side channel will have to implement the `SideChannel` abstract class and the following method.

 * `OnMessageReceived(IncomingMessage msg)` : You must implement this method and read the data from IncomingMessage.
  The data must be read in the order that it was written.

The side channel must also assign a `ChannelId` property in the constructor. The `ChannelId` is a Guid
(or UUID in Python) used to uniquely identify a side channel. This Guid must be the same on C# and Python.
There can only be one side channel of a certain id during communication.

To send data from C# to Python, create an `OutgoingMessage` instance, add data to it, call the
`base.QueueMessageToSend(msg)` method inside the side channel, and call the
`OutgoingMessage.Dispose()` method.

To register a side channel on the Unity side, call `Academy.Instance.RegisterSideChannel` with the side channel
as only argument.

### Python side
The side channel will have to implement the `SideChannel` abstract class. You must implement :

 * `on_message_received(self, msg: "IncomingMessage") -> None` : You must implement this method and read the data
  from IncomingMessage. The data must be read in the order that it was written.

The side channel must also assign a `channel_id` property in the constructor. The `channel_id` is a UUID
(referred in C# as Guid) used to uniquely identify a side channel. This number must be the same on C# and
Python. There can only be one side channel of a certain id during communication.

To assign the `channel_id` call the abstract class constructor with the appropriate `channel_id` as follows:

```python
super().__init__(my_channel_id)
```

To send a byte array from Python to C#, create an `OutgoingMessage` instance, add data to it, and call the
 `super().queue_message_to_send(msg)` method inside the side channel.

To register a side channel on the Python side, pass the side channel as argument when creating the
`UnityEnvironment` object. One of the arguments of the constructor (`side_channels`) is a list of side channels.

## Example implementation

Below is a simple implementation of a side channel that will exchange ascii encoded
strings between a Unity environment and Python.

### Example Unity C# code

The first step is to create the `StringLogSideChannel` class within the Unity project.
Here is an implementation of a `StringLogSideChannel` that will listen for messages
from python and print them to the Unity debug log, as well as send error messages
from Unity to python.

```csharp
using UnityEngine;
using MLAgents;
using MLAgents.SideChannels;
using System.Text;
using System;

public class StringLogSideChannel : SideChannel
{
    public StringLogSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    public override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        Debug.Log("From Python : " + receivedString);
    }

    public void SendDebugStatementToPython(string logString, string stackTrace, LogType type)
    {
        if (type == LogType.Error)
        {
            var stringToSend = type.ToString() + ": " + logString + "\n" + stackTrace;
            using (var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(stringToSend);
                QueueMessageToSend(msgOut);
            }
        }
    }
}
```

Once we have defined our custom side channel class, we need to ensure that it is
instantiated and registered. This can typically be done wherever the logic of
the side channel makes sense to be associated, for example on a MonoBehaviour
object that might need to access data from the side channel. Here we show a
simple MonoBehaviour object which instantiates and registeres the new side
channel. If you have not done it already, make sure that the MonoBehaviour
which registers the side channel is attached to a gameobject which will
be live in your Unity scene.

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

        // The channel must be registered with the Academy
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

### Example Python code

Now that we have created the necessary Unity C# classes, we can create their Python counterparts.

```python
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the StringLogChannel class
class StringLogChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)
```


We can then instantiate the new side channel,
launch a `UnityEnvironment` with that side channel active, and send a series of
messages to the Unity environment from Python using it.

```python
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

Now, if you run this script and press `Play` the Unity Editor when prompted,
the console in the Unity Editor will display a message at every Python step.
Additionally, if you press the Space Bar in the Unity Engine, a message will
appear in the terminal.
