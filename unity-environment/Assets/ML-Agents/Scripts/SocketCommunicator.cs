using Google.Protobuf;
using Grpc.Core;
using System.Net.Sockets;
using UnityEngine;


namespace MLAgents.Communicator
{

    public class SocketCommunicator : Communicator{

        const int messageLength = 12000;
        byte[] messageHolder = new byte[messageLength];
        int comPort;
        Socket sender;
        byte[] lengthHolder = new byte[4];
        CommunicatorParameters communicatorParameters;

        UnityOutput defaultOutput = new UnityOutput();


        public SocketCommunicator(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;
            defaultOutput.Header = new Header
            {
                Status = 200
            };
        }

        public PythonParameters Initialize(AcademyParameters academyParameters,
                                           out UnityRLInput unityInput)
        {

            sender = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            sender.Connect("localhost", communicatorParameters.Port);

            PythonParameters pp = PythonParameters.Parser.ParseFrom(Receive());
            Send(academyParameters.ToByteArray());

            unityInput = UnityInput.Parser.ParseFrom(Receive()).RlInput;
            return pp;

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
                System.Buffer.BlockCopy(messageHolder, 0, result, location, fragment);
                location += fragment;

                //rMessageString.Append(Encoding.ASCII.GetString(messageHolder, 0, fragment));
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

        public UnityRLInput SendOuput(UnityRLOutput unityOutput)
        {
            defaultOutput.RlOutput = unityOutput;
            Send(defaultOutput.ToByteArray());

            return UnityInput.Parser.ParseFrom(Receive()).RlInput;
        }

        /// Ends connection and closes environment
        private void OnApplicationQuit()
        {

            //TODO

        }

    }
}
