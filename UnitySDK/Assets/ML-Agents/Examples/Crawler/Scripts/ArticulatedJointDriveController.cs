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
        public ArticulationBody arb;
        public Transform t;
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

            arb.xDrive = xDrive;
            arb.yDrive = yDrive;
            arb.zDrive = zDrive;
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
        /// Reset BodyPart list and dictionary.
        /// </summary>
        public void Reset()
        {
            bodyPartsDict.Clear();
            bodyPartsList.Clear();
        }
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
                t = t,
                startingPos = t.position,
                startingRot = t.rotation
            };
            
            
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
        
        
        public static Transform FindBodyPartByName(Transform rootBody, string bodyPartName)
        {
            Queue<Transform> queue = new Queue<Transform>();
            queue.Enqueue(rootBody);
            while (queue.Count > 0)
            {
                Transform child = queue.Dequeue();
                if (child.name == bodyPartName)
                    return child;
                foreach(Transform t in child)
                    queue.Enqueue(t);
            }
            return null;
        }
    }
}
