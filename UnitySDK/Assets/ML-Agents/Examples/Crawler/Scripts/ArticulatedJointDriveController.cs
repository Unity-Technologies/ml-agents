using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    /// <summary>
    /// Used to store relevant information for acting and learning for each body part in agent.
    /// </summary>
    [System.Serializable]
    public class ArticulationBodyPart
    {
        //[Header("Body Part Info")][Space(10)] public ConfigurableJoint joint;
        public ArticulationBody arb;
        [HideInInspector] public Vector3 startingPos;
        [HideInInspector] public Quaternion startingRot;

        [Header("Ground & Target Contact")][Space(10)]
        public GroundContact groundContact;

        public TargetContact targetContact;

        [FormerlySerializedAs("thisJDController")]
        [HideInInspector] public ArticulatedJointDriveController thisJdController;

        [Header("Current Joint Settings")][Space(10)]
        public Vector3 currentEularJointRotation;

        [HideInInspector] public float currentStrength;
        public float currentXNormalizedRot;
        public float currentYNormalizedRot;
        public float currentZNormalizedRot;

        [Header("Other Debug Info")][Space(10)]
        public Vector3 currentJointForce;

        public float currentJointForceSqrMag;
        public Vector3 currentJointTorque;
        public float currentJointTorqueSqrMag;
        public AnimationCurve jointForceCurve = new AnimationCurve();
        public AnimationCurve jointTorqueCurve = new AnimationCurve();

        /// <summary>
        /// Reset body part to initial configuration.
        /// </summary>
        public void Reset(ArticulationBodyPart bp)
        {
            bp.arb.transform.position = bp.startingPos;
            bp.arb.transform.rotation = bp.startingRot;
            
            // Can't assigned articulation body velocitys/angularVelocities
            //bp.arb.velocity = Vector3.zero;
            //bp.arb.angularVelocity = Vector3.zero;
            if (bp.groundContact)
            {
                bp.groundContact.touchingGround = false;
            }

            if (bp.targetContact)
            {
                bp.targetContact.touchingTarget = false;
            }
        }

        /// <summary>
        /// Apply torque according to defined goal `x, y, z` angle and force `strength`.
        /// </summary>
        public void SetJointTargetRotation(float x, float y, float z)
        {
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;

            var xDrive = arb.xDrive;
            var yDrive = arb.yDrive;
            var zDrive = arb.zDrive;
            
            

            var xRot = Mathf.Lerp(xDrive.lowerLimit, xDrive.upperLimit, x);
            var yRot = Mathf.Lerp(yDrive.lowerLimit, yDrive.upperLimit, y);
            var zRot = Mathf.Lerp(zDrive.lowerLimit, zDrive.upperLimit, z);

            currentXNormalizedRot =
                Mathf.InverseLerp(xDrive.lowerLimit, xDrive.upperLimit, xRot);
            
            currentYNormalizedRot = Mathf.InverseLerp(yDrive.lowerLimit, yDrive.upperLimit, yRot);
            currentZNormalizedRot = Mathf.InverseLerp(zDrive.lowerLimit, zDrive.upperLimit, zRot);

            //joint.targetRotation = Quaternion.Euler(xRot, yRot, zRot); // Original code
            xDrive.target = xRot; yDrive.target = yRot; zDrive.target = zRot;

            arb.xDrive = xDrive; arb.yDrive = yDrive; arb.zDrive = zDrive;
            
            currentEularJointRotation = new Vector3(xRot, yRot, zRot);
        }

        public void SetJointStrength(float strength)
        {
            var xDrive = arb.xDrive;
            var yDrive = arb.yDrive;
            var zDrive = arb.zDrive;
                
            var rawVal = (strength + 1f) * 0.5f * thisJdController.maxJointForceLimit;
            
            xDrive.stiffness = yDrive.stiffness = zDrive.stiffness = thisJdController.maxJointSpring;
            xDrive.damping = yDrive.damping = zDrive.damping = thisJdController.jointDampen;
            xDrive.forceLimit = yDrive.forceLimit = zDrive.forceLimit = rawVal;

            // Slerp drive does not exist, so we try to set strength for each axis individually
            arb.xDrive = xDrive;
            arb.yDrive = yDrive;
            arb.zDrive = zDrive;
            //joint.slerpDrive = jd;
            currentStrength = rawVal;
        }
    }

    public class ArticulatedJointDriveController : MonoBehaviour
    {
        [Header("Joint Drive Settings")][Space(10)]
        public float maxJointSpring;

        public float jointDampen;
        public float maxJointForceLimit;
        float m_FacingDot;

        [HideInInspector] public Dictionary<Transform, ArticulationBodyPart> bodyPartsDict = new Dictionary<Transform, ArticulationBodyPart>();

        [HideInInspector] public List<ArticulationBodyPart> bodyPartsList = new List<ArticulationBodyPart>();

        /// <summary>
        /// Create BodyPart object and add it to dictionary.
        /// </summary>
        public void SetupBodyPart(Transform t)
        {
            // Either parent(in case of legs) or game object itself(in case of body) must have ArticulationBody
            var arb = t.GetComponent<ArticulationBody>();
            if (arb == null)
                arb = t.parent.GetComponent<ArticulationBody>();
            
            var bp = new ArticulationBodyPart()
            {
                arb =  arb,
                startingPos = t.position,
                startingRot = t.rotation
            };
            
            // Does not exist in articulation body
            //bp.rb.maxAngularVelocity = 100;

            // Add & setup the ground contact script
            bp.groundContact = t.GetComponent<GroundContact>();
            if (!bp.groundContact)
            {
                bp.groundContact = t.gameObject.AddComponent<GroundContact>();
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }
            else
            {
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }

            // Add & setup the target contact script
            bp.targetContact = t.GetComponent<TargetContact>();
            if (!bp.targetContact)
            {
                bp.targetContact = t.gameObject.AddComponent<TargetContact>();
            }

            bp.thisJdController = this;
            bodyPartsDict.Add(t, bp);
            bodyPartsList.Add(bp);
        }

        public void GetCurrentJointForces()
        {
            /*
            foreach (var bodyPart in bodyPartsDict.Values)
            {
                if (!bodyPart.arb.isRoot)
                {
                    // Why do we need a force here ?
                    //bodyPart.currentJointForce = bodyPart.arb;
                    bodyPart.currentJointForceSqrMag = bodyPart.joint.currentForce.magnitude;
                    bodyPart.currentJointTorque = bodyPart.joint.currentTorque;
                    bodyPart.currentJointTorqueSqrMag = bodyPart.joint.currentTorque.magnitude;
                    if (Application.isEditor)
                    {
                        if (bodyPart.jointForceCurve.length > 1000)
                        {
                            bodyPart.jointForceCurve = new AnimationCurve();
                        }

                        if (bodyPart.jointTorqueCurve.length > 1000)
                        {
                            bodyPart.jointTorqueCurve = new AnimationCurve();
                        }

                        bodyPart.jointForceCurve.AddKey(Time.time, bodyPart.currentJointForceSqrMag);
                        bodyPart.jointTorqueCurve.AddKey(Time.time, bodyPart.currentJointTorqueSqrMag);
                    }
                }
            }
            */
        }
    }
}
