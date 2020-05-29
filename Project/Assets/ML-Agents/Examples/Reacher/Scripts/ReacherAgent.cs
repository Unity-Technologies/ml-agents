using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgentsExamples
{



    public class ReacherAgent : Agent
    {
        public GameObject pendulumA;
        public GameObject pendulumB;
        public GameObject hand;
        public GameObject goal;

        float m_GoalDegree;
        private Rigidbody m_RbA;
        private Rigidbody m_RbB;

        // speed of the goal zone around the arm (in radians)
        private float m_GoalSpeed;

        // radius of the goal zone
        private float m_GoalSize;

        // Magnitude of sinusoidal (cosine) deviation of the goal along the vertical dimension
        private float m_Deviation;

        // Frequency of the cosine deviation of the goal along the vertical dimension
        private float m_DeviationFreq;

        EnvironmentParameters m_ResetParams;

        /// <summary>
        /// Collect the rigidbodies of the reacher in order to resue them for
        /// observations and actions.
        /// </summary>
        public override void Initialize()
        {
            m_RbA = pendulumA.GetComponent<Rigidbody>();
            m_RbB = pendulumB.GetComponent<Rigidbody>();

            m_ResetParams = Academy.Instance.EnvironmentParameters;

            SetResetParameters();
        }

        /// <summary>
        /// We collect the normalized rotations, angularal velocities, and velocities of both
        /// limbs of the reacher as well as the relative position of the target and hand.
        /// </summary>
        public override void CollectObservations(VectorSensor vectorSensor)
        {
            vectorSensor.AddObservation(pendulumA.transform.localPosition);
            vectorSensor.AddObservation(pendulumA.transform.rotation);
            vectorSensor.AddObservation(m_RbA.angularVelocity);
            vectorSensor.AddObservation(m_RbA.velocity);

            vectorSensor.AddObservation(pendulumB.transform.localPosition);
            vectorSensor.AddObservation(pendulumB.transform.rotation);
            vectorSensor.AddObservation(m_RbB.angularVelocity);
            vectorSensor.AddObservation(m_RbB.velocity);

            vectorSensor.AddObservation(goal.transform.localPosition);
            vectorSensor.AddObservation(hand.transform.localPosition);

            vectorSensor.AddObservation(m_GoalSpeed);
        }

        /// <summary>
        /// The agent's four actions correspond to torques on each of the two joints.
        /// </summary>
        public override void OnActionReceived(float[] vectorAction)
        {
            m_GoalDegree += m_GoalSpeed;
            UpdateGoalPosition();

            var torqueX = Mathf.Clamp(vectorAction[0], -1f, 1f) * 150f;
            var torqueZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * 150f;
            m_RbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));

            torqueX = Mathf.Clamp(vectorAction[2], -1f, 1f) * 150f;
            torqueZ = Mathf.Clamp(vectorAction[3], -1f, 1f) * 150f;
            m_RbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));
        }

        /// <summary>
        /// Used to move the position of the target goal around the agent.
        /// </summary>
        void UpdateGoalPosition()
        {
            var radians = m_GoalDegree * Mathf.PI / 180f;
            var goalX = 8f * Mathf.Cos(radians);
            var goalY = 8f * Mathf.Sin(radians);
            var goalZ = m_Deviation * Mathf.Cos(m_DeviationFreq * radians);
            goal.transform.position = new Vector3(goalY, goalZ, goalX) + transform.position;
        }

        /// <summary>
        /// Resets the position and velocity of the agent and the goal.
        /// </summary>
        public override void OnEpisodeBegin()
        {
            pendulumA.transform.position = new Vector3(0f, -4f, 0f) + transform.position;
            pendulumA.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
            m_RbA.velocity = Vector3.zero;
            m_RbA.angularVelocity = Vector3.zero;

            pendulumB.transform.position = new Vector3(0f, -10f, 0f) + transform.position;
            pendulumB.transform.rotation = Quaternion.Euler(180f, 0f, 0f);
            m_RbB.velocity = Vector3.zero;
            m_RbB.angularVelocity = Vector3.zero;

            m_GoalDegree = Random.Range(0, 360);
            UpdateGoalPosition();

            SetResetParameters();


            goal.transform.localScale = new Vector3(m_GoalSize, m_GoalSize, m_GoalSize);
        }

        public void SetResetParameters()
        {
            m_GoalSize = m_ResetParams.GetWithDefault("goal_size", 5);
            m_GoalSpeed = Random.Range(-1f, 1f) * m_ResetParams.GetWithDefault("goal_speed", 1);
            m_Deviation = m_ResetParams.GetWithDefault("deviation", 0);
            m_DeviationFreq = m_ResetParams.GetWithDefault("deviation_freq", 0);
        }

        public override void Heuristic(float[] actionsOut)
        {
            for (var i = 0; i < actionsOut.Length; i++)
            {
                actionsOut[i] = Random.Range(-1f, 1f);
            }
        }
    }
}