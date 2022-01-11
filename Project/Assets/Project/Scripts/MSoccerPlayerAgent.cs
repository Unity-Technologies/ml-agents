using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
namespace Project
{
    public class MSoccerPlayerAgent : Agent
    {
    #region Refenrence

        MSoccerEnvironment Environment;
        Rigidbody Rigidbody;

    #endregion

    #region Config

        public int team_id;
        Vector3 spawn_position;
        Quaternion spawn_rotation;

    #endregion

    #region AgentEvent

        public override void Initialize()
        {
            Environment = GetComponentInParent<MSoccerEnvironment>();
            Rigidbody = GetComponent<Rigidbody>();
            var transform1 = transform;
            spawn_position = transform1.position;
            spawn_rotation = transform1.rotation;

        }
        public override void Heuristic(in ActionBuffers actionsOut)
        {

        }
        public override void CollectObservations(VectorSensor sensor) { base.CollectObservations( sensor ); }
        public override void OnActionReceived(ActionBuffers actions) { base.OnActionReceived( actions ); }
        public override void OnEpisodeBegin()
        {
            var pos = spawn_position +
                      Vector3.right * Random.Range( -5f, 5f );
            var rot = spawn_rotation *
                      Quaternion.Euler( 0, Random.Range( -10f, 10f ), 0 );
            transform.SetPositionAndRotation( pos, rot );
            Rigidbody.velocity = Vector3.zero;
            Rigidbody.angularVelocity = Vector3.zero;
        }

    #endregion

    #region Interface

    #endregion
    }
}
