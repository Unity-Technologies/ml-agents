using Google.Protobuf;
using Grpc.Core;
using System.Net.Sockets;
using UnityEngine;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{

    public class SocketCommunicator : Communicator
    {

        const int messageLength = 12000;
        byte[] messageHolder = new byte[messageLength];
        int comPort;
        Socket sender;
        byte[] lengthHolder = new byte[4];
        CommunicatorParameters communicatorParameters;


        public SocketCommunicator(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;
        }

        public UnityInput Initialize(UnityOutput unityOutput,
                                     out UnityInput unityInput)
        {

            sender = new Socket(
                AddressFamily.InterNetwork,
                SocketType.Stream,
                ProtocolType.Tcp);
            sender.Connect("localhost", communicatorParameters.port);

            UnityInput initializationInput =
                UnityInput.Parser.ParseFrom(Receive());

            Send(WrapMessage(unityOutput, 200).ToByteArray());

            unityInput = UnityInput.Parser.ParseFrom(Receive());
            return initializationInput;

        }

        byte[] Receive()
        {
            sender.Receive(lengthHolder);
            int totalLength = System.BitConverter.ToInt32(lengthHolder, 0);
            int location = 0;
            byte[] result = new byte[totalLength];
            while (location != totalLength)
            {
                int fragment = sender.Receive(messageHolder);
                System.Buffer.BlockCopy(
                    messageHolder, 0, result, location, fragment);
                location += fragment;
            }
            return result;
        }

        void Send(byte[] input)
        {
            byte[] newArray = new byte[input.Length + 4];
            input.CopyTo(newArray, 4);
            System.BitConverter.GetBytes(input.Length).CopyTo(newArray, 0);
            sender.Send(newArray);
        }

        public void Close()
        {
            // TODO: Implement
        }

        public UnityInput Exchange(UnityOutput unityOutput)
        {
            Send(WrapMessage(unityOutput, 200).ToByteArray());

            return UnityInput.Parser.ParseFrom(Receive());
        }

        UnityMessage WrapMessage(UnityOutput content, int status)
        {
            return new UnityMessage
            {
                Header = new Header { Status = status },
                UnityOutput = content
            };
        }

        /// Ends connection and closes environment
        private void OnApplicationQuit()
        {
            //TODO
        }

    }
}
