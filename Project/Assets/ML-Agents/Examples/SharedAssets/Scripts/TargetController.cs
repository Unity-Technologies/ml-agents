using UnityEngine;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using UnityEngine.Events;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// Utility class to allow target placement and collision detection with an agent
    /// Add this script to the target you want the agent to touch.
    /// Callbacks will be triggered any time the target is touched with a collider tagged as 'tagToDetect'
    /// </summary>
    public class TargetController : MonoBehaviour
    {
        public string tagToDetect;

        [Header("Target Placement")] [Space(10)]
        public bool moveTargetToRandomPosIfTouched; //Should the target respawn to a different position when touched

        public float targetSpawnRadius; //The radius in which a target can be randomly spawned.
        private Vector3 m_startingPos; //the starting position of the target
        private Agent m_agentTouching; //the agent currently touching the target

        [System.Serializable]
        public class TriggerEvent : UnityEvent<Collider>
        {
        }

        [Header("Trigger Callbacks")] public bool triggerIsTouching;
        public TriggerEvent onTriggerEnterEvent = new TriggerEvent();
        public TriggerEvent onTriggerStayEvent = new TriggerEvent();
        public TriggerEvent onTriggerExitEvent = new TriggerEvent();

        [System.Serializable]
        public class CollisionEvent : UnityEvent<Collision>
        {
        }

        [Header("Collision Callbacks")] public bool colliderIsTouching;
        public CollisionEvent onCollisionEnterEvent = new CollisionEvent();
        public CollisionEvent onCollisionStayEvent = new CollisionEvent();
        public CollisionEvent onCollisionExitEvent = new CollisionEvent();

        // Start is called before the first frame update
        void OnEnable()
        {
            m_startingPos = transform.position;
            if (moveTargetToRandomPosIfTouched)
            {
                MoveTargetToRandomPosition();
            }
        }

        /// <summary>
        /// Moves target to a random position within specified radius.
        /// </summary>
        public void MoveTargetToRandomPosition()
        {
            var newTargetPos = m_startingPos + (Random.insideUnitSphere * targetSpawnRadius);
            newTargetPos.y = 5;
            transform.position = newTargetPos;
        }

        private void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(tagToDetect))
            {
                colliderIsTouching = true;
                onCollisionEnterEvent.Invoke(col);
                if (moveTargetToRandomPosIfTouched)
                {
                    MoveTargetToRandomPosition();
                }
            }
        }

        private void OnCollisionStay(Collision col)
        {
            if (col.transform.CompareTag(tagToDetect))
            {
                colliderIsTouching = true;
                onCollisionStayEvent.Invoke(col);
            }
        }

        private void OnCollisionExit(Collision col)
        {
            if (col.transform.CompareTag(tagToDetect))
            {
                colliderIsTouching = false;
                onCollisionExitEvent.Invoke(col);
            }
        }

        private void OnTriggerEnter(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                triggerIsTouching = true;
                onTriggerEnterEvent.Invoke(col);
            }
        }

        private void OnTriggerStay(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                triggerIsTouching = true;
                onTriggerStayEvent.Invoke(col);
            }
        }

        private void OnTriggerExit(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                triggerIsTouching = false;
                onTriggerExitEvent.Invoke(col);
            }
        }
    }
}
