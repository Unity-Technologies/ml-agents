using NUnit.Framework;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using Unity.PerformanceTesting;
using UnityEngine;

namespace MLAgentsExamples.Tests.Performance
{
    [TestFixture]
    public class SensorPerformanceTests
    {
        string[] s_Markers =
        {
            "root.InitializeSensors",
            "root.AgentSendState.CollectObservations",
            "root.AgentSendState.RequestDecision"
        };
        const int k_NumAgentSteps = 10;
        const int k_MeasurementCount = 25;
        const int k_MarkerTestSteps = 10;

        [SetUp]
        public void SetUp()
        {
            // Step a dummy agent here, so that we don't time the Academy initialization connection attempt and
            // any other static setup costs.
            RunAgent<DummyAgent>(1, 0, ObservableAttributeOptions.ExamineAll);
        }

        /// <summary>
        /// Simple Agent just used for "burning in" the Academy for testing.
        /// </summary>
        class DummyAgent : Agent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
            }

            public override void Heuristic(in ActionBuffers actionsOut)
            {
            }
        }

        /// <summary>
        /// Agent used for performance testing that uses the CollectObservations interface.
        /// </summary>
        class CollectObservationsAgent : Agent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
                sensor.AddObservation(new Vector3(1, 2, 3));
                sensor.AddObservation(new Quaternion(1, 2, 3, 4));
            }

            public override void Heuristic(in ActionBuffers actionsOut)
            {
            }
        }

        /// <summary>
        /// Agent used for performance testing that uses the ObservableAttributes on fields.
        /// </summary>
        class ObservableFieldAgent : Agent
        {
            [Observable]
            public Vector3 Vector3Field = new Vector3(1, 2, 3);

            [Observable]
            public Quaternion QuaternionField = new Quaternion(1, 2, 3, 4);

            public override void Heuristic(in ActionBuffers actionsOut)
            {
            }
        }

        /// <summary>
        /// Agent used for performance testing that uses the ObservableAttributes on properties.
        /// </summary>
        class ObservablePropertyAgent : Agent
        {
            Vector3 m_Vector3Field = new Vector3(1, 2, 3);

            [Observable]
            Vector3 Vector3Property
            {
                get { return m_Vector3Field; }
            }

            Quaternion m_QuaternionField = new Quaternion(1, 2, 3, 4);

            [Observable]
            Quaternion QuaternionProperty
            {
                get { return m_QuaternionField; }
            }

            public override void Heuristic(in ActionBuffers actionsOut)
            {
            }
        }

        void RunAgent<T>(int numSteps, int obsSize, ObservableAttributeOptions obsOptions) where T : Agent
        {
            var agentGameObj = new GameObject();
            var agent = agentGameObj.AddComponent<T>();

            var behaviorParams = agent.GetComponent<BehaviorParameters>();
            behaviorParams.BrainParameters.VectorObservationSize = obsSize;
            behaviorParams.ObservableAttributeHandling = obsOptions;

            agent.LazyInitialize();
            for (var i = 0; i < numSteps; i++)
            {
                agent.RequestDecision();
                Academy.Instance.EnvironmentStep();
            }
            Object.DestroyImmediate(agentGameObj);
        }

        [Test, Performance]
        public void TestCollectObservationsAgent()
        {
            Measure.Method(() =>
            {
                RunAgent<CollectObservationsAgent>(k_NumAgentSteps, 7, ObservableAttributeOptions.Ignore);
            })
                .MeasurementCount(k_MeasurementCount)
                .GC()
                .Run();
        }

        [Test, Performance]
        public void TestObservableFieldAgent()
        {
            Measure.Method(() =>
            {
                RunAgent<ObservableFieldAgent>(k_NumAgentSteps, 0, ObservableAttributeOptions.ExcludeInherited);
            })
                .MeasurementCount(k_MeasurementCount)
                .GC()
                .Run();
        }

        [Test, Performance]
        public void TestObservablePropertyAgent()
        {
            Measure.Method(() =>
                {
                    RunAgent<ObservablePropertyAgent>(k_NumAgentSteps, 0, ObservableAttributeOptions.ExcludeInherited);
                })
                .MeasurementCount(k_MeasurementCount)
                .GC()
                .Run();
        }

        [Test, Performance]
        public void TestCollectObservationsAgentMarkers()
        {
            using (Measure.ProfilerMarkers(s_Markers))
            {
                for (var i = 0; i < k_MarkerTestSteps; i++)
                {
                    RunAgent<CollectObservationsAgent>(k_NumAgentSteps, 7, ObservableAttributeOptions.Ignore);
                }
            }
        }

        [Test, Performance]
        public void TestObservableFieldAgentMarkers()
        {
            using (Measure.ProfilerMarkers(s_Markers))
            {
                for (var i = 0; i < k_MarkerTestSteps; i++)
                {
                    RunAgent<ObservableFieldAgent>(k_NumAgentSteps, 0, ObservableAttributeOptions.ExcludeInherited);
                }
            }
        }

        [Test, Performance]
        public void TestObservablePropertyAgentMarkers()
        {
            using (Measure.ProfilerMarkers(s_Markers))
            {
                for (var i = 0; i < k_MarkerTestSteps; i++)
                {
                    RunAgent<ObservableFieldAgent>(k_NumAgentSteps, 0, ObservableAttributeOptions.ExcludeInherited);
                }
            }
        }
    }
}
