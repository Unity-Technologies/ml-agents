using System.Collections;


public abstract class SideChannel
{

    public Queue MessageQueue = new Queue();
    protected abstract int ChannelType();

    protected abstract void OnMessageReceived(byte[] data);

    protected void QueueMessageToSend(byte[] data)
    {
        MessageQueue.Enqueue(data);
    }

}