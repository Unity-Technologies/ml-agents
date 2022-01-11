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
        public int position_id;
        PositionConfig m_position_config;
        Vector3 spawn_position;
        Quaternion spawn_rotation;

    #endregion

    #region AgentEvent

        public override void Initialize()
        {
            Environment = GetComponentInParent<MSoccerEnvironment>();
            Rigidbody = GetComponent<Rigidbody>();
            m_position_config = Environment.positions[position_id];
            var transform1 = transform;
            spawn_position = transform1.position;
            spawn_rotation = transform1.rotation;

        }

        void OnCollisionEnter(Collision collision)
        {
            var collided = collision.gameObject;
            if (!collided.CompareTag( "ball" ))
            {
                return;
            }
            AddReward( Environment.player_touch_ball_reward );
            var force =
                Environment.player_kick_power * m_position_config.kick_power_scale *
                Mathf.Clamp( Vector3.Dot( Rigidbody.velocity, transform.forward ), 0, 10 );
            var direction = (collision.contacts[0].point - transform.position).normalized;
            collision.rigidbody.AddForce( direction * force );

        }
        public override void Heuristic(in ActionBuffers actionsOut) { }
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

    #region Util

        public void Move(int z_axis_move, int x_axis_move, int y_axis_rot)
        {
            var transform1 = transform;
            var x_vector = m_position_config.lateral_speed_scale * transform1.right * (x_axis_move - 1);
            var z_vector = m_position_config.forward_speed_scale * transform1.forward * (z_axis_move - 1);
            var y_vector = transform1.up * (y_axis_rot - 1);
            transform1.Rotate( y_vector, Time.deltaTime * Environment.player_base_angular_speed );
            Rigidbody.AddForce( (x_vector + z_vector) * Environment.player_base_speed, ForceMode.VelocityChange );
        }

    #endregion

    }
}
