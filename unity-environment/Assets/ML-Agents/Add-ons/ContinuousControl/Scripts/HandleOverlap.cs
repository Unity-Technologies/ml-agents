using UnityEngine;

namespace MLAgents
{
	public class HandleOverlap : MonoBehaviour {
        public GameObject Parent;
        
        /// <summary>
        /// OnCollisionEnter is called when this collider/rigidbody has begun
        /// touching another rigidbody/collider.
        /// </summary>
        /// <param name="other">The Collision data associated with this collision.</param>
        void OnCollisionEnter(Collision other)
        {
            Collider myCollider = GetComponent<Collider>();
            if (myCollider == null)
                return;
            // only ingore if part of the same object
            if (myCollider.transform.root != other.transform.root)
                return;
            Physics.IgnoreCollision(myCollider, other.collider);
        }        
    }
}