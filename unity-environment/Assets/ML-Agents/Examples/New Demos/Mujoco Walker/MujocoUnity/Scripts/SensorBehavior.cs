using UnityEngine;

namespace MujocoUnity
{
    public class SensorBehavior : MonoBehaviour
    {
        MujocoController _mujocoController;
        Collider _collider;
        void Start ()
        {
            _mujocoController = GetComponentInParent<MujocoController>();
            _collider = GetComponent<Collider>();
        }
        void OnCollisionEnter(Collision other) 
        {
            if (_mujocoController!=null)
                _mujocoController.SensorCollisionEnter(_collider, other);
        }
        void OnCollisionExit(Collision other) 
        {
            if (_mujocoController!=null)
                _mujocoController.SensorCollisionExit(_collider, other);
        }

    }
}