using UnityEngine;
using MLAgents;

public class ArticulatedReacherAgent : Agent
{
    public GameObject reacherRoot;
    public GameObject pendulumA;
    public GameObject pendulumB;
    public GameObject hand;
    public GameObject goal;
    public GameObject reacherRootPrefab;
    
    private ReacherAcademy m_MyAcademy;
    float m_GoalDegree;

    private string m_PendulumAName;
    private string m_PendulumBName;
    private string m_ReacherRootName;
    private string m_HandName;
    
    private ArticulationBody m_RbA;
    private ArticulationBody m_RbB;
    // speed of the goal zone around the arm (in radians)
    private float m_GoalSpeed;
    // radius of the goal zone
    private float m_GoalSize;
    // Magnitude of sinusoidal (cosine) deviation of the goal along the vertical dimension
    private float m_Deviation;
    // Frequency of the cosine deviation of the goal along the vertical dimension
    private float m_DeviationFreq;

    /// <summary>
    /// Collect the rigidbodies of the reacher in order to resue them for
    /// observations and actions.
    /// </summary>
    public override void InitializeAgent()
    {
        m_RbA = pendulumA.GetComponent<ArticulationBody>();
        m_RbB = pendulumB.GetComponent<ArticulationBody>();
        m_MyAcademy = GameObject.Find("Academy").GetComponent<ReacherAcademy>();

        
        m_PendulumAName = pendulumA.name;
        m_PendulumBName = pendulumB.name; 
        m_ReacherRootName = reacherRoot.name;
        m_HandName = hand.name;
        
        SetResetParameters();
    }

    /// <summary>
    /// We collect the normalized rotations, angularal velocities, and velocities of both
    /// limbs of the reacher as well as the relative position of the target and hand.
    /// </summary>
    public override void CollectObservations()
    {
        Vector3 pendulumAPosToLocalSpace = transform.InverseTransformPoint(pendulumA.transform.position); 
        AddVectorObs(pendulumAPosToLocalSpace);
        AddVectorObs(pendulumA.transform.rotation);
        // Below resulted in 1.691 after 1 M steps
        AddVectorObs(transform.InverseTransformVector(m_RbA.angularVelocity));
        AddVectorObs(transform.InverseTransformVector(m_RbA.velocity));
        // Below resulted in 0.0732 after 1 M steps, not learning
        //AddVectorObs(m_RbA.angularVelocity);
        //AddVectorObs(m_RbA.velocity);

        
        
        Vector3 pendulumBPosToLocalSpace = transform.InverseTransformPoint(pendulumB.transform.position);
        AddVectorObs(pendulumBPosToLocalSpace);
        AddVectorObs(pendulumB.transform.rotation);
        
        // Below resulted in 1.691 after 1 M steps
        AddVectorObs(transform.InverseTransformVector(m_RbB.angularVelocity));
        AddVectorObs(transform.InverseTransformVector(m_RbB.velocity));
        // Below resulted in 0.0732 after 1 M steps, not learning
        //AddVectorObs(m_RbB.angularVelocity);
        //AddVectorObs(m_RbB.velocity);

        Vector3 goalPosToLocalSpace = transform.InverseTransformPoint(goal.transform.position); 
        AddVectorObs(goalPosToLocalSpace);

        Vector3 handPosToLocalSpace = transform.InverseTransformPoint(hand.transform.position); 
        AddVectorObs(handPosToLocalSpace);

        //AddVectorObs(m_GoalSpeed);
        // Below resulted in 4.18 after 1 M steps and reached 37.52 after 1.25 M steps
        AddVectorObs(Vector3.Distance(goalPosToLocalSpace, handPosToLocalSpace));
    }
    
    

    /// <summary>
    /// The agent's four actions correspond to torques on each of the two joints.
    /// </summary>
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        m_GoalDegree += m_GoalSpeed;
        UpdateGoalPosition();

        //float maxTorque = 150f;
        float maxTorque = 150f;
        
        var torqueX = Mathf.Clamp(vectorAction[0], -1f, 1f) * maxTorque;
        var torqueZ = Mathf.Clamp(vectorAction[1], -1f, 1f) * maxTorque;
        m_RbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));
        
        torqueX = Mathf.Clamp(vectorAction[2], -1f, 1f) * maxTorque;
        torqueZ = Mathf.Clamp(vectorAction[3], -1f, 1f) * maxTorque;
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
    public override void AgentReset()
    {
        Vector3 position = reacherRoot.transform.position;
        Quaternion rotation = Quaternion.identity;
        
        DestroyImmediate(reacherRoot);
        reacherRoot = Instantiate(reacherRootPrefab, position, rotation);
        reacherRoot.transform.parent = transform;
        reacherRoot.name = m_ReacherRootName;

        pendulumA = reacherRoot.transform.GetChild(0).Find(m_PendulumAName).gameObject;
        pendulumB = pendulumA.transform.Find(m_PendulumBName).gameObject;
        hand = pendulumB.transform.GetChild(0).Find(m_HandName).gameObject;
        
        
        m_RbA = pendulumA.GetComponent<ArticulationBody>();
        m_RbB = pendulumB.GetComponent<ArticulationBody>();
        
        m_GoalDegree = Random.Range(0, 360);
        UpdateGoalPosition();

        SetResetParameters();

        goal.transform.localScale = new Vector3(m_GoalSize, m_GoalSize, m_GoalSize);
        
        // Supply correct newly instantiated hand for collision checking
        ArticulatedReacherGoal gc = goal.GetComponent<ArticulatedReacherGoal>();
        gc.hand = hand;
    }
    

    public void SetResetParameters()
    {
        m_GoalSize = m_MyAcademy.resetParameters["goal_size"];
        m_GoalSpeed = Random.Range(-1f, 1f) * m_MyAcademy.resetParameters["goal_speed"];
        m_Deviation = m_MyAcademy.resetParameters["deviation"];
        m_DeviationFreq = m_MyAcademy.resetParameters["deviation_freq"];
    }
    
}
