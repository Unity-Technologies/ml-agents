using UnityEngine;

namespace MujocoUnity
{
    public class SensorBehavior : MonoBehaviour
    {
        MujocoAgent _mujocoAgent;
        Collider _collider;
        void Start ()
        {
            _mujocoAgent = GetComponentInParent<MujocoAgent>();
            _collider = GetComponent<Collider>();
        }
        void OnCollisionEnter(Collision other) 
        {
            if (_mujocoAgent!=null)
                _mujocoAgent.SensorCollisionEnter(_collider, other);
        }
        void OnCollisionExit(Collision other) 
        {
            if (_mujocoAgent!=null)
                _mujocoAgent.SensorCollisionExit(_collider, other);
        }

    }
}