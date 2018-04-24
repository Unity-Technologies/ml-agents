using Grpc.Core;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace MLAgents.Communicator
{
    /// Responsible for communication with Python API.
    public class RpcCommunicator2 : Communicator
    {

        UnityToPython.UnityToPythonClient client;

        CommunicatorParameters communicatorParameters;

        public RpcCommunicator2(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;

        }

        public PythonParameters Initialize(AcademyParameters academyParameters,
                                           out UnityRLInput unityInput)
        {
            
            Channel channel = new Channel("localhost:"+communicatorParameters.Port, ChannelCredentials.Insecure);

            client = new UnityToPython.UnityToPythonClient(channel);
            UnityInitializationOutput initOutput = new UnityInitializationOutput();
            initOutput.Header = new Header { Status = 200 };
            initOutput.AcademyParameters = academyParameters;
            var result = client.Initialize(initOutput).PythonParameters;
            UnityOutput output = new UnityOutput();
            output.Header = new Header { Status = 200 };
            unityInput = client.Send(output).RlInput;

            EditorApplication.playmodeStateChanged = HandleOnPlayModeChanged;

            return result;
        }

        public void Close()
        {
            UnityOutput output = new UnityOutput();
            output.Header = new Header { Status = 400 };
            client.Send(output); 
        }

        public UnityRLInput SendOuput(UnityRLOutput unityOutput)
        {
            UnityOutput output = new UnityOutput();
            output.Header = new Header { Status = 200 };
            output.RlOutput = unityOutput;
            return client.Send(output).RlInput; 
        }

        /// Ends connection and closes environment
        private void OnApplicationQuit()
        {
            Close();
        }

        void HandleOnPlayModeChanged()
        {
            // This method is run whenever the playmode state is changed.
            if (!EditorApplication.isPlaying)
            {
                Close();
            }
        }

    }
}
