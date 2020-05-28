using NUnit.Framework;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using Unity.PerformanceTesting;
using UnityEngine;

namespace Unity.MLAgents.Tests.Performance
{
    [TestFixture]
    public class SensorPerformanceTests
    {
        [SetUp]
        public void SetUp()
        {
            // Run Academy initialization here, so that we don't time the connection attempt.
            Academy.Instance.LazyInitialize();
        }

        class CollectObservationsAgent : Agent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
                sensor.AddObservation(new Vector3(1, 2, 3));
                sensor.AddObservation(new Quaternion(1, 2, 3, 4));
            }

            public override void Heuristic(float[] actionsOut)
            {
            }
        }

        class ObservableFieldAgent : Agent
        {
            [Observable]
            public Vector3 Vector3Field = new Vector3(1, 2, 3);

            [Observable]
            public Quaternion QuaternionField = new Quaternion(1, 2, 3, 4);

            public override void Heuristic(float[] actionsOut)
            {
            }
        }

        void RunAgent<T>(int numSteps, int obsSize) where T : Agent
        {
            var agentGameObj = new GameObject();
            var agent = agentGameObj.AddComponent<T>();

            var decisionRequester = agent.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.Awake();

            var behaviorParams = agent.GetComponent<BehaviorParameters>();
            behaviorParams.BrainParameters.VectorObservationSize = obsSize;

            agent.LazyInitialize();
            for (var i = 0; i < numSteps; i++)
            {
                Academy.Instance.EnvironmentStep();
            }
            Object.DestroyImmediate(agentGameObj);
        }

        [Test, Performance]
        public void TestCollectObservationsAgent()
        {
            Measure.Method(() =>
            {
                RunAgent<CollectObservationsAgent>(10, 7);
            })
                .MeasurementCount(10)
                .Run();
        }

        [Test, Performance]
        public void TestObservableFieldAgent()
        {
            Measure.Method(() =>
            {
                RunAgent<ObservableFieldAgent>(10, 0);
            })
                .MeasurementCount(10)
                .Run();
        }

        [Test, Performance]
        public void TestCollectObservationsAgentMarkers()
        {
            string[] markers =
            {
                "root.InitializeSensors",
                "root.AgentSendState.CollectObservations"
            };
            using (Measure.ProfilerMarkers(markers))
            {
                RunAgent<CollectObservationsAgent>(10, 7);
            }
        }

        [Test, Performance]
        public void TestObservableFieldAgentMarkers()
        {
            string[] markers =
            {
                "root.InitializeSensors",
                "root.AgentSendState.CollectObservations"
            };

            using (Measure.ProfilerMarkers(markers))
            {
                RunAgent<ObservableFieldAgent>(10, 0);
            }
        }
    }
}
