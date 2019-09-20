using Google.Protobuf;
using System.Net.Sockets;
using UnityEngine;
using MLAgents.CommunicatorObjects;
using System.Threading.Tasks;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents
{
    public class SocketCommunicator : ICommunicator
    {
        private const float k_TimeOut = 10f;
        private const int k_MessageLength = 12000;
        byte[] m_MessageHolder = new byte[k_MessageLength];
        int m_ComPort;
        Socket m_Sender;
        byte[] m_LengthHolder = new byte[4];
        CommunicatorParameters m_CommunicatorParameters;


        public SocketCommunicator(CommunicatorParameters communicatorParameters)
        {
            m_CommunicatorParameters = communicatorParameters;
        }

        /// <summary>
        /// Initialize the communicator by sending the first UnityOutput and receiving the
        /// first UnityInput. The second UnityInput is stored in the unityInput argument.
        /// </summary>
        /// <returns>The first Unity Input.</returns>
        /// <param name="unityOutput">The first Unity Output.</param>
        /// <param name="unityInput">The second Unity input.</param>
        public UnityInput Initialize(UnityOutput unityOutput,
            out UnityInput unityInput)
        {
            m_Sender = new Socket(
                AddressFamily.InterNetwork,
                SocketType.Stream,
                ProtocolType.Tcp);
            m_Sender.Connect("localhost", m_CommunicatorParameters.port);

            var initializationInput =
                UnityMessage.Parser.ParseFrom(Receive());

            Send(WrapMessage(unityOutput, 200).ToByteArray());

            unityInput = UnityMessage.Parser.ParseFrom(Receive()).UnityInput;
#if UNITY_EDITOR
#if UNITY_2017_2_OR_NEWER
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#else
            EditorApplication.playmodeStateChanged += HandleOnPlayModeChanged;
#endif
#endif
            return initializationInput.UnityInput;
        }

        /// <summary>
        /// Uses the socke to receive a byte[] from External. Reassemble a message that was split
        /// by External if it was too long.
        /// </summary>
        /// <returns>The byte[] sent by External.</returns>
        byte[] Receive()
        {
            m_Sender.Receive(m_LengthHolder);
            var totalLength = System.BitConverter.ToInt32(m_LengthHolder, 0);
            var location = 0;
            var result = new byte[totalLength];
            while (location != totalLength)
            {
                var fragment = m_Sender.Receive(m_MessageHolder);
                System.Buffer.BlockCopy(
                    m_MessageHolder, 0, result, location, fragment);
                location += fragment;
            }
            return result;
        }

        /// <summary>
        /// Send the specified input via socket to External. Split the message into smaller
        /// parts if it is too long.
        /// </summary>
        /// <param name="input">The byte[] to be sent.</param>
        void Send(byte[] input)
        {
            var newArray = new byte[input.Length + 4];
            input.CopyTo(newArray, 4);
            System.BitConverter.GetBytes(input.Length).CopyTo(newArray, 0);
            m_Sender.Send(newArray);
        }

        /// <summary>
        /// Close the communicator gracefully on both sides of the communication.
        /// </summary>
        public void Close()
        {
            Send(WrapMessage(null, 400).ToByteArray());
        }

        /// <summary>
        /// Send a UnityOutput and receives a UnityInput.
        /// </summary>
        /// <returns>The next UnityInput.</returns>
        /// <param name="unityOutput">The UnityOutput to be sent.</param>
        public UnityInput Exchange(UnityOutput unityOutput)
        {
            Send(WrapMessage(unityOutput, 200).ToByteArray());
            byte[] received = null;
            var task = Task.Run(() => received = Receive());
            if (!task.Wait(System.TimeSpan.FromSeconds(k_TimeOut)))
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

        /// <summary>
        /// Wraps the UnityOuptut into a message with the appropriate status.
        /// </summary>
        /// <returns>The UnityMessage corresponding.</returns>
        /// <param name="content">The UnityOutput to be wrapped.</param>
        /// <param name="status">The status of the message.</param>
        private static UnityMessage WrapMessage(UnityOutput content, int status)
        {
            return new UnityMessage
            {
                Header = new Header { Status = status },
                UnityOutput = content
            };
        }

        /// <summary>
        /// When the Unity application quits, the communicator must be closed
        /// </summary>
        private void OnApplicationQuit()
        {
            Close();
        }

#if UNITY_EDITOR
#if UNITY_2017_2_OR_NEWER
        /// <summary>
        /// When the editor exits, the communicator must be closed
        /// </summary>
        /// <param name="state">State.</param>
        private void HandleOnPlayModeChanged(PlayModeStateChange state)
        {
            // This method is run whenever the playmode state is changed.
            if (state == PlayModeStateChange.ExitingPlayMode)
            {
                Close();
            }
        }

#else
        /// <summary>
        /// When the editor exits, the communicator must be closed
        /// </summary>
        private void HandleOnPlayModeChanged()
        {
            // This method is run whenever the playmode state is changed.
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Close();
            }
        }

#endif
#endif
    }
}
