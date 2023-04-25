// using System;
// using System.Collections;
// using UnityEngine;
// using Unity.MLAgents;
// using Unity.MLAgents.Actuators;
// using Unity.MLAgentsExamples;
// using Unity.MLAgents.Sensors;
// using UnityEngine.Serialization;
// using Random = UnityEngine.Random;
//
// [RequireComponent(typeof(JointDriveController))] // Required to set joint forces
// public class CrawlerAgentNew: Agent
// {
//
//     [Header("Walk Speed")]
//     [Range(0.1f, m_maxWalkingSpeed)]
//     [SerializeField]
//     [Tooltip(
//         "The speed the agent will try to match.\n\n" +
//         "TRAINING:\n" +
//         "For VariableSpeed envs, this value will randomize at the start of each training episode.\n" +
//         "Otherwise the agent will try to match the speed set here.\n\n" +
//         "INFERENCE:\n" +
//         "During inference, VariableSpeed agents will modify their behavior based on this value " +
//         "whereas the CrawlerDynamic & CrawlerStatic agents will run at the speed specified during training "
//     )]
//     //The walking speed to try and achieve
//     private float m_TargetWalkingSpeed = m_maxWalkingSpeed;
//
//     const float m_maxWalkingSpeed = 15; //The max walking speed
//
//     //The current target walking speed. Clamped because a value of zero will cause NaNs
//     public float TargetWalkingSpeed
//     {
//         get { return m_TargetWalkingSpeed; }
//         set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
//     }
//
//     //The direction an agent will walk during training.
//     [Header("Target To Walk Towards")]
//     // public Transform TargetPrefab; //Target prefab to use in Dynamic envs
//     public Rigidbody m_Target; //Target the agent will walk towards during training.
//     public Vector3 targetStartingPos; //The starting position of the target
//     public float targetSpawnRadius; //The radius in which a target can be randomly spawned.
//
//     public void Start()
//     {
//         throw new NotImplementedException();
//     }
//
//     [Header("Body Parts")][Space(10)] public Transform body;
//     public Transform leg0Upper;
//     [FormerlySerializedAs("leg0Lower")] public Transform leg0Middle;
//     public Transform leg0Lower;
//     public Transform leg1Upper;
//     [FormerlySerializedAs("leg1Lower")] public Transform leg1Middle;
//     public Transform leg1Lower;
//     public Transform leg2Upper;
//     [FormerlySerializedAs("leg2Lower")] public Transform leg2Middle;
//     public Transform leg2Lower;
//     public Transform leg3Upper;
//     [FormerlySerializedAs("leg3Lower")] public Transform leg3Middle;
//     public Transform leg3Lower;
//
//     //This will be used as a stabilized model space reference point for observations
//     //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
//     // OrientationCubeController m_OrientationCube;
//     public Transform VirtualRoot;
//
//     //The indicator graphic gameobject that points towards the target
//     DirectionIndicator m_DirectionIndicator;
//     JointDriveController m_JdController;
//
//     [Header("Foot Grounded Visualization")]
//     [Space(10)]
//     public bool useFootGroundedVisualization;
//     // public MeshRenderer foot0;
//     // public MeshRenderer foot1;
//     // public MeshRenderer foot2;
//     // public MeshRenderer foot3;
//     public Transform foot0;
//     public Transform foot1;
//     public Transform foot2;
//     public Transform foot3;
//     public Material groundedMaterial;
//     public Material unGroundedMaterial;
//
//     [Header("REWARD CURVES")]
//     public AnimationCurve lookDirRewardCurve;
//     public AnimationCurve positionRewardCurve;
//
//     [Header("PROCEDURAL TERRAIN")] public ProceduralGround procGenGround;
//     public override void Initialize()
//     {
//         // SpawnTarget(TargetPrefab, transform.position); //spawn target
//
//         // m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
//         m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
//         m_JdController = GetComponent<JointDriveController>();
//
//         //Setup each body part
//         m_JdController.SetupBodyPart(body);
//         m_JdController.SetupBodyPart(leg0Upper);
//         m_JdController.SetupBodyPart(leg0Middle);
//         m_JdController.SetupBodyPart(leg1Upper);
//         m_JdController.SetupBodyPart(leg1Middle);
//         m_JdController.SetupBodyPart(leg2Upper);
//         m_JdController.SetupBodyPart(leg2Middle);
//         m_JdController.SetupBodyPart(leg3Upper);
//         m_JdController.SetupBodyPart(leg3Middle);
//         m_JdController.SetupBodyPart(leg0Lower);
//         m_JdController.SetupBodyPart(leg1Lower);
//         m_JdController.SetupBodyPart(leg2Lower);
//         m_JdController.SetupBodyPart(leg3Lower);
//
//         targetStartingPos = m_Target.position;
//     }
//
//     /// <summary>
//     /// Spawns a target prefab at pos
//     /// </summary>
//     /// <param name="prefab"></param>
//     /// <param name="pos"></param>
//     void SpawnTarget(Transform prefab, Vector3 pos)
//     {
//         var obj = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
//         obj.localScale = Vector3.one * 0.25f;
//         m_Target = obj.GetComponent<Rigidbody>();
//     }
//
//     float sinTimer = 0f;
//
//     void ResetTarget()
//     {
//         print("Resetting target");
//         m_Target.velocity = Vector3.zero;
//         m_Target.angularVelocity = Vector3.zero;
//         m_Target.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
//         m_Target.position = targetStartingPos + new Vector3(Random.Range(-targetSpawnRadius, targetSpawnRadius), 0, Random.Range(-targetSpawnRadius, targetSpawnRadius));
//     }
//
//     /// <summary>
//     /// Loop over body parts and reset them to initial conditions.
//     /// </summary>
//     public override void OnEpisodeBegin()
//     {
//         procGenGround.Generate();
//
//         //Reset our target
//         ResetTarget();
//
//         UpdateOrientationObjects();
//         sinTimer = 0;
//         foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
//         {
//             bodyPart.Reset(bodyPart);
//             bodyPart.InitializeRandomJointSettings();
//         }
//         body.position += new Vector3(Random.Range(-5f, 5f), 0, Random.Range(-5f, 5f));
//         //Random start rotation to help generalize
//         body.rotation = Quaternion.Euler(Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f), Random.Range(0.0f, 360.0f));
//
//
//         //Set our goal walking speed
//         TargetWalkingSpeed = Random.Range(0.1f, m_maxWalkingSpeed);
//         StartCoroutine(WaitingPeriod());
//     }
//
//     public bool canRequestDecision = false;
//     IEnumerator WaitingPeriod()
//     {
//         canRequestDecision = false;
//         yield return new WaitForSeconds(1);
//         WaitForFixedUpdate wait = new WaitForFixedUpdate();
//         //wait until the body rigidbody is not moving
//         var bodyRB = m_JdController.bodyPartsDict[body].rb;
//         while (bodyRB.velocity.magnitude > 0.1f && bodyRB.angularVelocity.magnitude > 0.1f)
//         {
//             yield return wait;
//         }
//         canRequestDecision = true;
//     }
//     /// <summary>
//     /// Add relevant information on each body part to observations.
//     /// </summary>
//     public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
//     {
//         //GROUND CHECK
//         sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground
//
//         //56
//         // sensor.AddObservation(VirtualRoot.InverseTransformDirection(bp.rb.angularVelocity));
//         // sensor.AddObservation(VirtualRoot.InverseTransformDirection(bp.rb.velocity));
//         if (bp.rb.transform != body)
//         {
//             //24
//             // sensor.AddObservation(VirtualRoot.InverseTransformPoint(bp.rb.position));
//             sensor.AddObservation(bp.rb.transform.localRotation); //32
//             // sensor.AddObservation(bp.rb.transform.localRotation.eulerAngles); //32
//             // sensor.AddObservation(VirtualRoot.InverseTransformDirection(bp.rb.transform.localRotation.eulerAngles)); //32
//             sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit); //8
//         }
//     }
//
//     public LayerMask groundLayer;
//     float GetDistToGround(Transform t)
//     {
//         RaycastHit hit;
//         var maxDist = 2.0f;
//         Ray ray = new Ray(t.position, Vector3.down);
//         if (Physics.Raycast(ray, out hit, maxDist, groundLayer))
//         {
//             return hit.distance / maxDist;
//         }
//         else
//         {
//             return 1f;
//         }
//     }
//
//     public Vector3 dirToTarget;
//     /// <summary>
//     /// Loop over body parts to add them to observation.
//     /// </summary>
//     public override void CollectObservations(VectorSensor sensor)
//     {
//
//         sensor.AddObservation(sinTimer);
//         sinTimer += Time.fixedDeltaTime;
//
//
//
//
//         // var cubeForward = VirtualRoot.transform.forward;
//
//         //distance to ground normalized
//         sensor.AddObservation(GetDistToGround(foot0.transform));
//         sensor.AddObservation(GetDistToGround(foot1.transform));
//         sensor.AddObservation(GetDistToGround(foot2.transform));
//         sensor.AddObservation(GetDistToGround(foot3.transform));
//         sensor.AddObservation(GetDistToGround(body.transform));
//
//         sensor.AddObservation(VirtualRoot.InverseTransformPoint(foot0.transform.position));
//         sensor.AddObservation(VirtualRoot.InverseTransformPoint(foot1.transform.position));
//         sensor.AddObservation(VirtualRoot.InverseTransformPoint(foot2.transform.position));
//         sensor.AddObservation(VirtualRoot.InverseTransformPoint(foot3.transform.position));
//         // sensor.AddObservation(body.InverseTransformPoint(foot0.transform.position));
//         // sensor.AddObservation(body.InverseTransformPoint(foot1.transform.position));
//         // sensor.AddObservation(body.InverseTransformPoint(foot2.transform.position));
//         // sensor.AddObservation(body.InverseTransformPoint(foot3.transform.position));
//
//
//         //direction we should face
//         // sensor.AddObservation(Quaternion.LookRotation(lookDir, Vector3.up));
//
//
//         sensor.AddObservation(VirtualRoot.InverseTransformDirection(m_JdController.bodyPartsDict[body].rb.velocity));
//         sensor.AddObservation(VirtualRoot.InverseTransformDirection(m_JdController.bodyPartsDict[body].rb.angularVelocity));
//
//
//         //rotation delta
//         // sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));
//         // // sensor.AddObservation(Quaternion.Dot(body.rotation, VirtualRoot.rotation));
//         // sensor.AddObservation((Quaternion.Dot(body.rotation, VirtualRoot.rotation) + 1) * .5F);
//         // // sensor.AddObservation((Vector3.Dot(body.up, Vector3.up) + 1) * .5F);
//
//
//         sensor.AddObservation(VirtualRoot.InverseTransformDirection(body.up));
//         sensor.AddObservation(VirtualRoot.InverseTransformDirection(body.forward));
//
//         var up = (Vector3.Dot(VirtualRoot.up, body.up) + 1) * .5F;
//         var upRew = positionRewardCurve.Evaluate(1 - up);
//         var forward = (Vector3.Dot(VirtualRoot.forward, body.forward) + 1) * .5F;
//         var forwardRew = positionRewardCurve.Evaluate(1 - forward);
//         sensor.AddObservation(up);
//         sensor.AddObservation(forward);
//         // sensor.AddObservation((up * forward)/2);
//
//         // sensor.AddObservation(upRew);
//         // sensor.AddObservation(forwardRew);
//         // sensor.AddObservation((upRew * forwardRew)/2);
//
//
//         var pos = Mathf.Clamp(dirToTarget.magnitude, 0, 5) / 5;
//         var posRew = positionRewardCurve.Evaluate(pos);
//         sensor.AddObservation(pos);
//         // sensor.AddObservation(posRew);
//         // sensor.AddObservation((up * forward * posRew)/2);
//         // sensor.AddObservation((up * forward * pos)/2);
//
//         // sensor.AddObservation(posRew);
//         // sensor.AddObservation((upRew * forwardRew * posRew)/2);
//
//         //body ang vel
//         // //direction we should walk
//         // var moveVector = Vector3.ClampMagnitude(dirToTarget, 1.0f);
//         // sensor.AddObservation(moveVector);
//         // sensor.AddObservation(body.InverseTransformDirection(moveVector));
//
//         // var dirToWalk = Vector3.ProjectOnPlane(dirToTarget, Vector3.up).normalized;
//         //body rotation
//         // sensor.AddObservation(body.localRotation);
//         // sensor.AddObservation(body.localRotation);
//
//         //27 observations
//
//
//         // // //velocity we want to match
//         // // var velGoal = cubeForward * TargetWalkingSpeed;
//         // //ragdoll's avg vel
//         // var avgVel = GetAvgVelocity();
//         // //
//         // //current ragdoll velocity. normalized
//         // // sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
//         // sensor.AddObservation(Vector3.Dot(body.up, Vector3.up));
//         // sensor.AddObservation(Vector3.Dot(dirToTarget.normalized, avgVel));
//         // //avg body vel relative to cube
//         // // sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
//         // sensor.AddObservation(VirtualRoot.InverseTransformDirection(avgVel));
//         // //vel goal relative to cube
//         // sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
//         //
//         //Add pos of target relative to orientation cube
//         var targetPos = VirtualRoot.transform.InverseTransformPoint(m_Target.transform.position);
//         targetPos = Vector3.ClampMagnitude(targetPos, 10.0f);
//         sensor.AddObservation(targetPos);
//
//         // RaycastHit hit;
//         // float maxRaycastDist = 10;
//         // if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
//         // {
//         //     sensor.AddObservation(hit.distance / maxRaycastDist);
//         // }
//         // else
//         //     sensor.AddObservation(1);
//
//         foreach (var bodyPart in m_JdController.bodyPartsList)
//         {
//             CollectObservationBodyPart(bodyPart, sensor);
//         }
//
//     }
//     // public int decNum = 0;
//     public override void OnActionReceived(ActionBuffers actionBuffers)
//     {
//         // The dictionary with all the body parts in it are in the jdController
//         var bpDict = m_JdController.bodyPartsDict;
//
//         var continuousActions = actionBuffers.ContinuousActions;
//         var i = -1;
//         // Pick a new target joint rotation
//         bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
//         bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
//         bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
//         bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
//         bpDict[leg0Middle].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg1Middle].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg2Middle].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg3Middle].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
//         bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
//
//         // Update joint strength
//         bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
//         bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
//         bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
//         bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
//         bpDict[leg0Middle].SetJointStrength(continuousActions[++i]);
//         bpDict[leg1Middle].SetJointStrength(continuousActions[++i]);
//         bpDict[leg2Middle].SetJointStrength(continuousActions[++i]);
//         bpDict[leg3Middle].SetJointStrength(continuousActions[++i]);
//         bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
//         bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
//         bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
//         bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);
//         GiveRewards();
//     }
//
//     void FixedUpdate()
//     {
//
//         //if the target y pos is less than the body y pos, then we are falling
//         if (m_Target.transform.position.y < -10)
//         {
//             print("fell off");
//             ResetTarget();
//         }
//
//         dirToTarget = m_Target.transform.position - body.position;
//         dirToTarget.y = 0;
//
//         // m_Target.rotation *= Quaternion.AngleAxis(2f * Time.deltaTime, Vector3.up);
//
//         UpdateOrientationObjects();
//         if(canRequestDecision && Academy.Instance.StepCount % 5 == 0)
//         {
//             RequestDecision();
//         }
//
//         // // If enabled the feet will light up green when the foot is grounded.
//         // // This is just a visualization and isn't necessary for function
//         // if (useFootGroundedVisualization)
//         // {
//         //     foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
//         //         ? groundedMaterial
//         //         : unGroundedMaterial;
//         //     foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
//         //         ? groundedMaterial
//         //         : unGroundedMaterial;
//         //     foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
//         //         ? groundedMaterial
//         //         : unGroundedMaterial;
//         //     foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
//         //         ? groundedMaterial
//         //         : unGroundedMaterial;
//         // }
//
//     }
//
//     void GiveRewards()
//     {
//         // // var cubeForward = m_OrientationCube.transform.forward;
//         // //
//         // // Set reward for this step according to mixture of the following elements.
//         // // a. Match target speed
//         // //This reward will approach 1 if it matches perfectly and approach zero as it deviates
//         // // var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());
//         // var upReward = (Vector3.Dot(body.up, Vector3.up) + 1) * .5F;
//         // var moveReward = (Vector3.Dot(dirToTarget.normalized, GetAvgVelocity()) + 1) * .5F;
//         // // moveReward = Mathf.Clamp01(moveReward);
//         // // b. Rotation alignment with target direction.
//         // //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
//         // var lookAtTargetReward = (Vector3.Dot(dirToTarget, body.forward) + 1) * .5F;
//         // // var lookAtTargetReward = Vector3.Dot(dirToTarget.normalized, body.forward);
//         // // lookAtTargetReward = Mathf.Clamp01(lookAtTargetReward);
//         // // AddReward(matchSpeedReward * lookAtTargetReward);
//         // // AddReward(lookAtTargetReward/MaxStep);
//         // AddReward(moveReward * lookAtTargetReward * upReward);
//
//         // var lookAtTargetReward = (Quaternion.Dot(body.rotation, VirtualRoot.rotation) + 1) * .5F;
//         // AddReward(lookAtTargetReward);
//
//
//         // var upRew = (Vector3.Dot(body.up, Vector3.up) + 1) * .5F;
//         // AddReward(upRew);
//
//
//
//
//
//
//
//
//
//
//         // var moveRew = 1 - (Mathf.Clamp01(dirToTarget.magnitude / 10));
//         var posRewValRaw = Mathf.Clamp(dirToTarget.magnitude,0, 5) / 5;
//         var posRew = positionRewardCurve.Evaluate(posRewValRaw);
//
//
//         var up = (Vector3.Dot(VirtualRoot.up, body.up)+ 1) * .5F;
//         var forward = (Vector3.Dot(VirtualRoot.forward, body.forward)+ 1) * .5F;
//         var upRew = lookDirRewardCurve.Evaluate(1 - up);
//         var forwardRew = lookDirRewardCurve.Evaluate(1 - forward);
//
//         // var dirRew = (up * forward) / 2;
//         // var pRew = (up * forward * posRew)/2;
//
//
//         // var dirRew = (upRew * forwardRew) / 2;
//         var pRew = (upRew * forwardRew * posRew);
//         // var pRew = (posRew)/2;
//         // var totalRew = dirRew + pRew;
//         var totalRew = pRew;
//         AddReward(totalRew);
//
//
//
//
//
//
//
//
//
//
//
//     }
//
//     /// <summary>
//     /// Update OrientationCube and DirectionIndicator
//     /// </summary>
//     void UpdateOrientationObjects()
//     {
//         // var lookDir = dirToTarget.normalized;
//         var lookRotation = Quaternion.LookRotation(dirToTarget.normalized);
//         VirtualRoot.SetPositionAndRotation(body.position, lookRotation);
//
//         // VirtualRoot.SetPositionAndRotation(body.position, m_Target.transform.rotation);
//
//         // m_OrientationCube.UpdateOrientation(body, m_Target);
//         if (m_DirectionIndicator)
//         {
//             m_DirectionIndicator.transform.SetPositionAndRotation(VirtualRoot.position, Quaternion.LookRotation(dirToTarget));
//             // m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
//         }
//     }
//
//     /// <summary>
//     ///Returns the average velocity of all of the body parts
//     ///Using the velocity of the body only has shown to result in more erratic movement from the limbs
//     ///Using the average helps prevent this erratic movement
//     /// </summary>
//     Vector3 GetAvgVelocity()
//     {
//         Vector3 velSum = Vector3.zero;
//         Vector3 avgVel = Vector3.zero;
//
//         //ALL RBS
//         int numOfRb = 0;
//         foreach (var item in m_JdController.bodyPartsList)
//         {
//             numOfRb++;
//             velSum += item.rb.velocity;
//         }
//
//         avgVel = velSum / numOfRb;
//         return avgVel;
//     }
//
//     /// <summary>
//     /// Normalized value of the difference in actual speed vs goal walking speed.
//     /// </summary>
//     public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
//     {
//         //distance between our actual velocity and goal velocity
//         var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);
//
//         //return the value on a declining sigmoid shaped curve that decays from 1 to 0
//         //This reward will approach 1 if it matches perfectly and approach zero as it deviates
//         return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
//     }
//
//     /// <summary>
//     /// Agent touched the target
//     /// </summary>
//     public void TouchedTarget()
//     {
//         AddReward(1f);
//     }
// }
