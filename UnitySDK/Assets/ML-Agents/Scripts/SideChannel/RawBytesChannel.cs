using System.Collections;

public class RawBytesChannel : SideChannel
{

    private Queue m_MessagesReceived = new Queue();

    protected override int ChannelType() { return 0; }

    protected override void OnMessageReceived(byte[] data)
    {
        m_MessagesReceived.Enqueue(data);
    }

    public void SendRawBytes(byte[] data)
    {
        QueueMessageToSend(data);
    }

    public byte[] ReceiveRawBytes()
    {
        return (byte[])m_MessagesReceived.Dequeue();
    }

    public int MessageReceivedCount()
    {
        return m_MessagesReceived.Count;
    }

}