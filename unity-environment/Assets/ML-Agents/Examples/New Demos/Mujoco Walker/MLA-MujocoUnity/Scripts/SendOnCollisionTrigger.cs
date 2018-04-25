using UnityEngine;

namespace MlaMujocoUnity {
    public class SendOnCollisionTrigger : MonoBehaviour {
		// Use this for initialization
		void Start () {
			
		}
		
		// Update is called once per frame
		void Update () {
			
		}

		void OnCollisionEnter(Collision other) {
			// Messenger.
			var otherGameobject = other.gameObject;
            var mujocoAgent = otherGameobject.GetComponentInParent<MujocoAgent>();
			// if (mujocoAgent?.Length > 0)
			if (mujocoAgent != null)
				mujocoAgent.OnTerrainCollision(otherGameobject, this.gameObject);
		}
    }
}