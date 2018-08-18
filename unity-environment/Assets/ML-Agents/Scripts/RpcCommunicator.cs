using Grpc.Core;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{
    /// Responsible for communication with External using gRPC.
    public class RPCCommunicator : Communicator
    {
        /// If true, the communication is active.
        bool m_isOpen;

        /// The Unity to External client. 
        UnityToExternal.UnityToExternalClient m_client;

        /// The communicator parameters sent at construction
        CommunicatorParameters m_communicatorParameters;

        /// <summary>
        /// Initializes a new instance of the RPCCommunicator class.
        /// </summary>
        /// <param name="communicatorParameters">Communicator parameters.</param>
        public RPCCommunicator(CommunicatorParameters communicatorParameters)
        {
            m_communicatorParameters = communicatorParameters;
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
            m_isOpen = true;
            var channel = new Channel(
                "localhost:"+m_communicatorParameters.port, 
                ChannelCredentials.Insecure);

            m_client = new UnityToExternal.UnityToExternalClient(channel);
            var result = m_client.Exchange(WrapMessage(unityOutput, 200));
            unityInput = m_client.Exchange(WrapMessage(null, 200)).UnityInput;
#if UNITY_EDITOR
#if UNITY_2017_2_OR_NEWER
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#else
            EditorApplication.playmodeStateChanged += HandleOnPlayModeChanged;
#endif
#endif
            return result.UnityInput;
        }

        /// <summary>
        /// Close the communicator gracefully on both sides of the communication.
        /// </summary>
        public void Close()
        {
            if (!m_isOpen)
            {
                return;
            }

            try
            {
                m_client.Exchange(WrapMessage(null, 400));
                m_isOpen = false;
            }
            catch
            {
                return;
            }
        }

        /// <summary>
        /// Send a UnityOutput and receives a UnityInput.
        /// </summary>
        /// <returns>The next UnityInput.</returns>
        /// <param name="unityOutput">The UnityOutput to be sent.</param>
        public UnityInput Exchange(UnityOutput unityOutput)
        {
            if (!m_isOpen)
            {
                return null;
            }
            try
            {
                var message = m_client.Exchange(WrapMessage(unityOutput, 200));
                if (message.Header.Status == 200)
                {
                    return message.UnityInput;
                }
                else
                {
                    m_isOpen = false;
                    return null;
                }
            }
            catch
            {
                m_isOpen = false;
                return null;
            }
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
            if (state==PlayModeStateChange.ExitingPlayMode)
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
