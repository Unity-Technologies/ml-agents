using Google.Protobuf;
using Grpc.Core;
using System.Net.Sockets;
using UnityEngine;
using MLAgents.CommunicatorObjects;
using System.Threading.Tasks;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents
{

    public class SocketCommunicator : Communicator
    {
        private const float TIMEOUT = 10f;
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

            UnityMessage initializationInput =
                UnityMessage.Parser.ParseFrom(Receive());

            Send(WrapMessage(unityOutput, 200).ToByteArray());

            unityInput = UnityMessage.Parser.ParseFrom(Receive()).UnityInput;
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#endif
            return initializationInput.UnityInput;

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
            Send(WrapMessage(null, 400).ToByteArray());
        }

        public UnityInput Exchange(UnityOutput unityOutput)
        {
            Send(WrapMessage(unityOutput, 200).ToByteArray());
            byte[] received = null;
            var task = Task.Run(() => received = Receive());
            if (!task.Wait(System.TimeSpan.FromSeconds(TIMEOUT)))
            {
                throw new UnityAgentsException(
                    "The communicator took too long to respond.");
            }

            var message = UnityMessage.Parser.ParseFrom(received);

            if (message.Header.Status != 200)
            {
                return null;
            }
            return message.UnityInput;
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
            Close();
        }

#if UNITY_EDITOR
        void HandleOnPlayModeChanged(PlayModeStateChange state)
        {
            // This method is run whenever the playmode state is changed.
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                Close();
            }
        }
#endif

    }
}
