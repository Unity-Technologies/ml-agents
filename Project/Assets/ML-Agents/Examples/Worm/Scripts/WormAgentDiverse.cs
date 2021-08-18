using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;

using System.IO;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class WormAgentDiverse : Agent
{
    const float m_minWalkingSpeed = 1; //The min walking speed to obtain reward

    [Header("Target Prefabs")] public Transform TargetPrefab; //Target prefab to use in Dynamic envs
    private Transform m_Target; //Target the agent will walk towards during training.

    [Header("Body Parts")] public Transform bodySegment0;
    public Transform bodySegment1;
    public Transform bodySegment2;
    public Transform bodySegment3;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    private Vector3 m_StartingPos; //starting position of the agent

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position); //spawn target

        m_StartingPos = bodySegment0.position;
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        UpdateOrientationObjects();

        //Setup each body part
        m_JdController.SetupBodyPart(bodySegment0);
        m_JdController.SetupBodyPart(bodySegment1);
        m_JdController.SetupBodyPart(bodySegment2);
        m_JdController.SetupBodyPart(bodySegment3);

        using (StreamWriter file = new StreamWriter("Worm.txt"))
        {
            file.WriteLine("behavior,seg0_xpos,seg0_ypos,seg0_zpos,seg0_xrot,seg0_yrot,seg0_zrot,seg1_xpos,seg1_ypos,seg1_zpos,seg1_xrot,seg1_yrot,seg1_zrot,seg2_xpos,seg2_ypos,seg2_zpos,seg2_xrot,seg2_yrot,seg2_zrot,seg3_xpos,seg3_ypos,seg3_zpos,seg3_xrot,seg3_yrot,seg3_zrot");
        }
    }

    /// <summary>
    /// Spawns a target prefab at pos
    /// </summary>
    /// <param name="prefab"></param>
    /// <param name="pos"></param>
    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        bodySegment0.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public string CollectObservationBodyPart(BodyPart bp, VectorSensor sensor, string line)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround ? 1 : 0); // Whether the bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));


        if (bp.rb.transform != bodySegment0)
        {
            //Get position relative to hips in the context of our orientation cube's space
            sensor.AddObservation(
                m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position));
            sensor.AddObservation(bp.rb.transform.localRotation);
            line += m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position).x.ToString() + ",";
            line += m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position).y.ToString() + ",";
            line += m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - bodySegment0.position).z.ToString() + ",";
            line += bp.rb.transform.localRotation.eulerAngles.x.ToString() + ",";
            line += bp.rb.transform.localRotation.eulerAngles.y.ToString() + ",";
            line += bp.rb.transform.localRotation.eulerAngles.z.ToString() + ",";
        }
        else
        {
            line += "0,0,0,";
            line += bp.rb.transform.localRotation.eulerAngles.x.ToString() + ",";
            line += bp.rb.transform.localRotation.eulerAngles.y.ToString() + ",";
            line += bp.rb.transform.localRotation.eulerAngles.z.ToString() + ",";
        }

        if (bp.joint)
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);

        return line;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        RaycastHit hit;
        float maxDist = 10;
        if (Physics.Raycast(bodySegment0.position, Vector3.down, out hit, maxDist))
        {
            sensor.AddObservation(hit.distance / maxDist);
        }
        else
            sensor.AddObservation(1);

        var cubeForward = m_OrientationCube.transform.forward;
        var velGoal = cubeForward;
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        sensor.AddObservation(Quaternion.Angle(m_OrientationCube.transform.rotation,
                                  m_JdController.bodyPartsDict[bodySegment0].rb.rotation) / 180);
        sensor.AddObservation(Quaternion.FromToRotation(bodySegment0.forward, cubeForward));

        //Add pos of target relative to orientation cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        string line = "";
        line += GetComponent<DiversitySamplerComponent>().DiscreteSetting.ToString() + ",";
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            line = CollectObservationBodyPart(bodyPart, sensor, line);
        }
        using (StreamWriter file = new StreamWriter("Worm.txt", append: true))
        {
            file.WriteLine(line);
        }
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var i = -1;
        var continuousActions = actionBuffers.ContinuousActions;
        // Pick a new target joint rotation
        bpDict[bodySegment0].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[bodySegment1].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[bodySegment2].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        // Update joint strength
        bpDict[bodySegment0].SetJointStrength(continuousActions[++i]);
        bpDict[bodySegment1].SetJointStrength(continuousActions[++i]);
        bpDict[bodySegment2].SetJointStrength(continuousActions[++i]);

        //Reset if Worm fell through floor;
        if (bodySegment0.position.y < m_StartingPos.y - 2)
        {
            EndEpisode();
        }
    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        var velReward =
            GetMatchingVelocityReward(m_OrientationCube.transform.forward,
                m_JdController.bodyPartsDict[bodySegment0].rb.velocity);
        AddReward(velReward);
    }

    /// <summary>
    /// Positive one if moving towards goal.
    /// </summary>
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        var velocityProjection = Vector3.Project(actualVelocity, velocityGoal);
        var direction = Vector3.Angle(velocityProjection, velocityGoal) == 0 ? 1 : -1;
        var speed = velocityProjection.magnitude * direction;
        return speed > m_minWalkingSpeed ? 0.1f : 0;
    }

    /// <summary>
    /// Update OrientationCube and DirectionIndicator
    /// </summary>
    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(bodySegment0, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }
}
