using System.Collections.Generic;


public abstract class SideChannel
{

    public List<byte[]> MessageQueue = new List<byte[]>(); // List
    public abstract int ChannelType();

    public abstract void OnMessageReceived(byte[] data);

    protected void QueueMessageToSend(byte[] data)
    {
        MessageQueue.Add(data);
    }

}