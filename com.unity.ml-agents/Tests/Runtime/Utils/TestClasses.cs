using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;

[assembly: InternalsVisibleTo("Unity.ML-Agents.Editor.Tests")]
[assembly: InternalsVisibleTo("Unity.ML-Agents.Runtime.Sensor.Tests")]
[assembly: InternalsVisibleTo("Unity.ML-Agents.Extensions.EditorTests")]

namespace Unity.MLAgents.Utils.Tests
{
    internal class TestPolicy : IPolicy
    {
        public Action OnRequestDecision;
        ObservationWriter m_ObsWriter = new ObservationWriter();
        static ActionSpec s_ActionSpec = ActionSpec.MakeContinuous(1);
        static ActionBuffers s_EmptyActionBuffers = new ActionBuffers(new float[1], Array.Empty<int>());

        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            foreach (var sensor in sensors)
            {
                sensor.GetObservationProto(m_ObsWriter);
            }
            OnRequestDecision?.Invoke();
        }

        public ref readonly ActionBuffers DecideAction() { return ref s_EmptyActionBuffers; }

        public void Dispose() { }
    }

    public class TestAgent : Agent
    {
        internal AgentInfo _Info
        {
            get
            {
                return (AgentInfo)typeof(Agent).GetField("m_Info", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
            }
            set
            {
                typeof(Agent).GetField("m_Info", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(this, value);
            }
        }

        internal void SetPolicy(IPolicy policy)
        {
            typeof(Agent).GetField("m_Brain", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(this, policy);
        }

        internal IPolicy GetPolicy()
        {
            return (IPolicy)typeof(Agent).GetField("m_Brain", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
        }

        public int initializeAgentCalls;
        public int collectObservationsCalls;
        public int collectObservationsCallsForEpisode;
        public int agentActionCalls;
        public int agentActionCallsForEpisode;
        public int agentOnEpisodeBeginCalls;
        public int heuristicCalls;
        public TestSensor sensor1;
        public TestSensor sensor2;

        [Observable("observableFloat")]
        public float observableFloat;

        public override void Initialize()
        {
            initializeAgentCalls += 1;

            // Add in some custom Sensors so we can confirm they get sorted as expected.
            sensor1 = new TestSensor("testsensor1");
            sensor2 = new TestSensor("testsensor2");
            sensor2.compressionType = SensorCompressionType.PNG;

            sensors.Add(sensor2);
            sensors.Add(sensor1);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            collectObservationsCalls += 1;
            collectObservationsCallsForEpisode += 1;
            sensor.AddObservation(collectObservationsCallsForEpisode);
        }

        public override void OnActionReceived(ActionBuffers buffers)
        {
            agentActionCalls += 1;
            agentActionCallsForEpisode += 1;
            AddReward(0.1f);
        }

        public override void OnEpisodeBegin()
        {
            agentOnEpisodeBeginCalls += 1;
            collectObservationsCallsForEpisode = 0;
            agentActionCallsForEpisode = 0;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var obs = GetObservations();
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = (int)obs[0];
            heuristicCalls++;
        }
    }

    public class TestSensor : ISensor
    {
        public string sensorName;
        public int numWriteCalls;
        public int numCompressedCalls;
        public int numResetCalls;
        public SensorCompressionType compressionType = SensorCompressionType.None;

        public TestSensor(string n)
        {
            sensorName = n;
        }

        public ObservationSpec GetObservationSpec()
        {
            return ObservationSpec.Vector(0);
        }

        public int Write(ObservationWriter writer)
        {
            numWriteCalls++;
            // No-op
            return 0;
        }

        public byte[] GetCompressedObservation()
        {
            numCompressedCalls++;
            return new byte[] { 0 };
        }

        public CompressionSpec GetCompressionSpec()
        {
            return new CompressionSpec(compressionType);
        }

        public string GetName()
        {
            return sensorName;
        }

        public void Update() { }

        public void Reset()
        {
            numResetCalls++;
        }
    }

    public class TestClasses
    {
    }
}
