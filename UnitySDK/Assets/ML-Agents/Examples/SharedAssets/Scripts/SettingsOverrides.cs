using UnityEngine;
using MLAgents;

public class SettingsOverride : MonoBehaviour
{
    // Original values
    Vector3 m_OriginalGravity;
    float m_OriginalFixedDeltaTime;
    float m_OriginalMaximumDeltaTime;
    int m_OriginalSolverIterations;
    int m_OriginalSolverVelocityIterations;

    public float gravityMultiplier = 1.0f;
    public float fixedDeltaTime = Time.fixedDeltaTime;
    public float maximumDeltaTime = Time.maximumDeltaTime;

    [Header("Advanced physics settings")]
    public int solverIterations = Physics.defaultSolverIterations;
    public int solverVelocityIterations = Physics.defaultSolverVelocityIterations;

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

        var academy = FindObjectOfType<Academy>();
        academy.LazyInitialization();

        academy.FloatProperties.RegisterCallback("gravity", f => { Physics.gravity = new Vector3(0, -f, 0); });
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
