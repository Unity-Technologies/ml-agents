using Grpc.Core;
using System.Threading;
using System.Threading.Tasks;

namespace MLAgents.Communicator
{
    /// Responsible for communication with Python API.
    public class RpcCommunicator : Communicator
    {

        public class PythonUnityImpl : PythonToUnity.PythonToUnityBase
        {

            const int TIMEOUT = 10;
            public AcademyParameters academyParameters;
            public PythonParameters pythonParameters;
            public UnityRLOutput outputs = new UnityRLOutput();
            public UnityRLInput inputs = new UnityRLInput();

            public Command command;

            private ManualResetEvent manualResetEvent_in
                = new ManualResetEvent(false);
            private ManualResetEvent manualResetEvent_out
                = new ManualResetEvent(false);
            UnityOutput messageOutput = new UnityOutput
            {
                Header = new Header
                {
                    Status = 200
                }
            };

            public void WaitForInput()
            {
                var task = Task.Run(() => manualResetEvent_in.WaitOne());
                if (!task.Wait(System.TimeSpan.FromSeconds(TIMEOUT)))
                {
                    throw new UnityAgentsException(
                        "The communicator took too long to respond.");
                }
                manualResetEvent_in.Reset();

            }
            public void InputReceived()
            {
                manualResetEvent_in.Set();
            }
            public void WaitForOutput()
            {
                var task = Task.Run(() => manualResetEvent_out.WaitOne());
                if (!task.Wait(System.TimeSpan.FromSeconds(TIMEOUT)))
                {
                    throw new UnityAgentsException(
                        "The Unity took too long to respond.");
                }
                manualResetEvent_out.Reset();
            }
            public void OutputReceived()
            {
                manualResetEvent_out.Set();
            }

            public override Task<AcademyParameters> Initialize(
                PythonParameters request, ServerCallContext context)
            {
                pythonParameters = request;
                InputReceived();
                WaitForOutput();
                return Task.FromResult(academyParameters);
            }

            public override Task<UnityOutput> Send(
                UnityInput request, ServerCallContext context)
            {
                if (request.Header.Status != 200)
                {
                    inputs = null;
                    InputReceived();
                    return Task.FromResult(messageOutput);
                }

                inputs = request.RlInput;
                InputReceived();
                WaitForOutput();

                messageOutput.RlOutput = outputs;
                return Task.FromResult(messageOutput);
            }

        }

        CommunicatorParameters communicatorParameters;

        PythonUnityImpl comm = new PythonUnityImpl();

        public RpcCommunicator(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;

        }

        public PythonParameters Initialize(AcademyParameters academyParameters,
                                           out UnityRLInput unityInput)
        {
            Server server = new Server
            {
                Services = { PythonToUnity.BindService(comm) },
                Ports = {
                    new ServerPort("localhost",
                    communicatorParameters.Port,
                    ServerCredentials.Insecure)
                }
            };
            server.Start();

            comm.academyParameters = academyParameters;

            comm.WaitForInput();
            comm.OutputReceived();

            comm.WaitForInput();
            unityInput = comm.inputs;
            return comm.pythonParameters;
        }

        public void Close()
        {
            // TODO: Implement
        }

        public UnityRLInput SendOuput(UnityRLOutput unityOutput)
        {
            comm.outputs = unityOutput;
            comm.OutputReceived();
            comm.WaitForInput();
            return comm.inputs;
        }

        /// Ends connection and closes environment
        private void OnApplicationQuit()
        {
            // TODO Implement
        }

    }
}
