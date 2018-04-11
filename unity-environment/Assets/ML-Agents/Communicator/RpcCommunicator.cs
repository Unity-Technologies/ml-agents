using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Newtonsoft.Json;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.IO;

using MLAgents;
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
            //http://dotnetpattern.com/threading-manualresetevent
            public ManualResetEvent manualResetEvent_comm = new ManualResetEvent(false);
            public ManualResetEvent manualResetEvent_main = new ManualResetEvent(false);
            //public bool isInitialized = false;
            public AcademyParameters academyParameters;
            public PythonParameters pythonParameters;
            public UnityRLOutput outputs = new UnityRLOutput();
            public UnityRLInput inputs = new UnityRLInput();

            public Command command = Command.Step;
            // TODO : Figure this out, This should be a default...

            private ManualResetEvent manualResetEvent_in = new ManualResetEvent(false);
            private ManualResetEvent manualResetEvent_out = new ManualResetEvent(false);
            UnityOutput messageOutput = new UnityOutput
            {
                Header = new Header
                {
                    Status = 200
                }
            };

            public void WaitForInput()
            {
                manualResetEvent_in.WaitOne();
                manualResetEvent_in.Reset();
            }
            public void InputReceived()
            {
                manualResetEvent_in.Set();
            }
            public void WaitForOutput()
            {
                manualResetEvent_out.WaitOne();
                manualResetEvent_out.Reset();
            }
            public void OutputReceived()
            {
                manualResetEvent_out.Set();
            }

            public override Task<AcademyParameters> Initialize(PythonParameters request, ServerCallContext context)
            {
                //Debug.Log("Received an Initialization");
                //while (!isInitialized)
                //{
                //    Debug.Log("Was Not initialized Waiting 0.5 seconds");
                //    System.Threading.Thread.Sleep(500);
                //}
                //Debug.Log("Just Started Initialize");
                //TODO : convert academyParameters to MLAgents.AcademyParameters
                pythonParameters = request;
                //manualResetEvent_comm.WaitOne();
                //manualResetEvent_comm.Reset();
                //manualResetEvent_main.Set();
                InputReceived();
                WaitForOutput();
                //Debug.Log("About to send AcademyParameters");
                return Task.FromResult(academyParameters);
            }

            public override Task<UnityOutput> Send(UnityInput request, ServerCallContext context)
            {
                //Debug.Log("Received a Reset");
                //while (!isInitialized)
                //{
                //    System.Threading.Thread.Sleep(5000);
                //}
                if (request.Header.Status != 200)
                {
                    inputs = null;
                    InputReceived();
                    return Task.FromResult(messageOutput);
                }

                inputs = request.RlInput;
                InputReceived();
                WaitForOutput();
                //command = ExternalCommand.RESET;
                //manualResetEvent_main.Set();
                //manualResetEvent_comm.
                //Debug.Log("Send Received inputs ( SendOuytputs should be called next");
                //manualResetEvent_comm.WaitOne();
                //manualResetEvent_comm.Reset();
                //manualResetEvent_main.Set();
                //TODO : convert academyParameters to MLAgents.AcademyParameters
                //Debug.Log("About to return the reset result");
                //Debug.Log("The inputs should have been used and new outputs generated");
                messageOutput.RlOutput = outputs;
                return Task.FromResult(messageOutput);
            }

        }

        CommunicatorParameters communicatorParameters;


        PythonUnityImpl comm = new PythonUnityImpl();



        StreamWriter logWriter;
        string logPath;
        //TODO : Job of the Academy ? 






        /// Consrtuctor for the External Communicator
        public RpcCommunicator(CommunicatorParameters communicatorParameters)
        {
            this.communicatorParameters = communicatorParameters;

        }

       
        // TODO from here

        /// Contains the logic for the initializtation of the socket.
        public PythonParameters Initialize(AcademyParameters academyParameters,
                                           out UnityRLInput unityInput)
        {

            //TODO : Reimplement
            Application.logMessageReceived += HandleLog;
            logPath = Path.GetFullPath(".") + "/unity-environment.log";
            logWriter = new StreamWriter(logPath, false);
            logWriter.WriteLine(System.DateTime.Now.ToString());
            logWriter.WriteLine(" ");
            logWriter.Close();

            //Debug.Log("Starting Server");

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

            //comm.isInitialized = true;

            //comm.manualResetEvent_comm.Set();
            ////Debug.Log("comm.manualResetEvent_comm.Set();");
            //comm.manualResetEvent_main.WaitOne();
            ////Debug.Log("comm.manualResetEvent_main.WaitOne();");
            //comm.manualResetEvent_main.Reset();
            ////Debug.Log("comm.manualResetEvent_main.Reset();");
            comm.WaitForInput();
            comm.OutputReceived();

            comm.WaitForInput();
            // TODO : You must receive an input at this point 
            unityInput = comm.inputs;
            return comm.pythonParameters;
        }

        public void Close()
        {
            // TODO: Implement
        }

        void HandleLog(string logString, string stackTrace, LogType type)
        {
            logWriter = new StreamWriter(logPath, true);
            logWriter.WriteLine(type.ToString());
            logWriter.WriteLine(logString);
            logWriter.WriteLine(stackTrace);
            logWriter.Close();
        }

        //public EnvironmentParameters GetEnvironmentParameters()
        //{
        //    return comm.inputs.EnvironmentParameters;
        //}

        //public Command GetCommand()
        //{
        //    return comm.command;
        //}

        public UnityRLInput SendOuput(UnityRLOutput unityOutput)
        {
            //Debug.Log("SendOutput is called");
            comm.outputs = unityOutput;
            comm.OutputReceived();
            comm.WaitForInput();
            //try
            //{
            //    Debug.Log(unityOutput.AgentInfos["Ball3DBrain"].Value[0].StoredVectorActions[0]);
            //    Debug.Log(comm.inputs.AgentActions["Ball3DBrain"].Value[0].VectorActions[0]);
            //}
            //catch { }
            //comm.manualResetEvent_comm.Set();
            ////Debug.Log("comm.manualResetEvent_comm.Set();");
            //comm.manualResetEvent_main.WaitOne();
            ////Debug.Log("comm.manualResetEvent_main.WaitOne();");
            //comm.manualResetEvent_main.Reset();
            //Debug.Log("comm.manualResetEvent_main.Reset();");



            //Debug.Log("Inputs given to Batcher");
            return comm.inputs;
        }

        ///// Used to read Python-provided environment parameters
        //private void ReadArgs()
        //{
        //    string[] args = System.Environment.GetCommandLineArgs();
        //    var inputPort = "";
        //    var inputSeed = "";
        //    for (int i = 0; i < args.Length; i++)
        //    {
        //        if (args[i] == "--port")
        //        {
        //            inputPort = args[i + 1];
        //        }
        //        if (args[i] == "--seed")
        //        {
        //            inputSeed = args[i + 1];
        //        }
        //    }
        //    comPort = int.Parse(inputPort);
        //    randomSeed = int.Parse(inputSeed);
        //    Random.InitState(randomSeed);
        //}


        /// Ends connection and closes environment
        private void OnApplicationQuit()
        {

            //TODO

        }

    }
}
