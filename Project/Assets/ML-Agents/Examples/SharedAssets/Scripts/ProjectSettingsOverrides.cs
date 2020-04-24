using UnityEngine;
using Unity.MLAgents;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// A helper class for the ML-Agents example scenes to override various
    /// global settings, and restore them afterwards.
    /// This can modify some Physics and time-stepping properties, so you
    /// shouldn't copy it into your project unless you know what you're doing.
    /// </summary>
    public class ProjectSettingsOverrides : MonoBehaviour
    {
        // Original values

        Vector3 m_OriginalGravity;
        float m_OriginalFixedDeltaTime;
        float m_OriginalMaximumDeltaTime;
        int m_OriginalSolverIterations;
        int m_OriginalSolverVelocityIterations;

        [Tooltip("Increase or decrease the scene gravity. Use ~3x to make things less floaty")]
        public float gravityMultiplier = 1.0f;

        [Header("Advanced physics settings")]
        [Tooltip("The interval in seconds at which physics and other fixed frame rate updates (like MonoBehaviour's FixedUpdate) are performed.")]
        public float fixedDeltaTime = .02f;
        [Tooltip("The maximum time a frame can take. Physics and other fixed frame rate updates (like MonoBehaviour's FixedUpdate) will be performed only for this duration of time per frame.")]
        public float maximumDeltaTime = 1.0f / 3.0f;
        [Tooltip("Determines how accurately Rigidbody joints and collision contacts are resolved. (default 6). Must be positive.")]
        public int solverIterations = 6;
        [Tooltip("Affects how accurately the Rigidbody joints and collision contacts are resolved. (default 1). Must be positive.")]
        public int solverVelocityIterations = 1;

        public void Awake()
        {
            // Save the original values
            m_OriginalGravity = Physics.gravity;
            m_OriginalFixedDeltaTime = Time.fixedDeltaTime;
            m_OriginalMaximumDeltaTime = Time.maximumDeltaTime;
            m_OriginalSolverIterations = Physics.defaultSolverIterations;
            m_OriginalSolverVelocityIterations = Physics.defaultSolverVelocityIterations;

            // Override
            Physics.gravity *= gravityMultiplier;
            Time.fixedDeltaTime = fixedDeltaTime;
            Time.maximumDeltaTime = maximumDeltaTime;
            Physics.defaultSolverIterations = solverIterations;
            Physics.defaultSolverVelocityIterations = solverVelocityIterations;

            // Make sure the Academy singleton is initialized first, since it will create the SideChannels.
            Academy.Instance.EnvironmentParameters.RegisterCallback("gravity", f => { Physics.gravity = new Vector3(0, -f, 0); });
        }

        public void OnDestroy()
        {
            Physics.gravity = m_OriginalGravity;
            Time.fixedDeltaTime = m_OriginalFixedDeltaTime;
            Time.maximumDeltaTime = m_OriginalMaximumDeltaTime;
            Physics.defaultSolverIterations = m_OriginalSolverIterations;
            Physics.defaultSolverVelocityIterations = m_OriginalSolverVelocityIterations;
        }
    }
}
