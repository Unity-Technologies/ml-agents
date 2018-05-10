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
    /// Responsible for communication with Python API.
    public class RPCCommunicator : Communicator
    {

        UnityToExternal.UnityToExternalClient client;

        CommunicatorParameters communicatorParameters;

        public RPCCommunicator(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;

        }

        public UnityInput Initialize(UnityOutput unityOutput,
                                     out UnityInput unityInput)
        {
            
            Channel channel = new Channel(
                "localhost:"+communicatorParameters.port, 
                ChannelCredentials.Insecure);

            client = new UnityToExternal.UnityToExternalClient(channel);
            //var initOutput = new UnityInitializationOutput();
            //initOutput.Header = new Header { Status = 200 };
            //initOutput.AcademyParameters = academyParameters;
            var result = client.Exchange(WrapMessage(unityOutput, 200));
            //UnityOutput output = new UnityOutput();
            //output.Header = new Header { Status = 200 };
            unityInput = client.Exchange(WrapMessage(null, 200)).UnityInput;
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += HandleOnPlayModeChanged;
#endif
            return result.UnityInput;
        }

        public void Close()
        {
            try
            {
                //var output = new UnityOutput();
                //output.Header = new Header { Status = 400 };
                //client.Send(output);
                client.Exchange(WrapMessage(null, 400));
            }
            catch
            {
                return;
            }
        }

        public UnityInput Exchange(UnityOutput unityOutput)
        {
            //var output = new UnityOutput();
            //output.Header = new Header { Status = 200 };
            //output.RlOutput = unityOutput;
            try
            {
                return client.Exchange(WrapMessage(unityOutput, 200)).UnityInput;
            }
            catch
            {
                return null;
            }
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
            if (state==PlayModeStateChange.ExitingPlayMode)
            {
                Close();
            }
        }
#endif

    }
}
