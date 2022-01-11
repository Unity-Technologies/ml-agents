using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;
namespace Project
{
    public class MSoccerPlayerAgent : Agent
    {
    #region Refenrence

        MSoccerEnvironment Environment;
        BehaviorParameters Parameters;
        Rigidbody Rigidbody;

    #endregion

    #region Config

        [HideInInspector]
        public int team_id;
        public int position_id;
        PositionConfig m_position_config;
        Vector3 spawn_position;
        Quaternion spawn_rotation;

    #endregion

    #region AgentEvent

        public override void Initialize()
        {
            Parameters = GetComponent<BehaviorParameters>();
            Environment = GetComponentInParent<MSoccerEnvironment>();
            team_id = Parameters.TeamId;
            Rigidbody = GetComponent<Rigidbody>();
            Rigidbody.maxAngularVelocity = 500;
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
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var acts = actionsOut.DiscreteActions;
            //forward
            if (Input.GetKey( KeyCode.W ))
            {
                acts[0] = 1;
            }
            if (Input.GetKey( KeyCode.S ))
            {
                acts[0] = 2;
            }

            //rotate
            if (Input.GetKey( KeyCode.E ))
            {
                acts[2] = 1;
            }
            if (Input.GetKey( KeyCode.Q ))
            {
                acts[2] = 2;
            }

            //right
            if (Input.GetKey( KeyCode.D ))
            {
                acts[1] = 1;
            }
            if (Input.GetKey( KeyCode.A ))
            {
                acts[1] = 2;
            }
        }
        // public override void CollectObservations(VectorSensor sensor) { }
        public override void OnActionReceived(ActionBuffers actionsBuffers)
        {
            AddReward( m_position_config.timer_reward *
                       Environment.cur_step_ratio );

            var acts = actionsBuffers.DiscreteActions;

            Move(
                b_table[acts[0]],
                b_table[acts[1]],
                b_table[acts[2]] );
        }
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

        /// <summary>
        ///     ori_action_branch_table: old branch number -> new action
        /// </summary>
        int[] b_table =
        {
            0,
            1,
            -1
        };

        public void Move(int z_axis_move, int x_axis_move, int y_axis_rot)
        {
            var transform1 = transform;
            var z_vector = m_position_config.forward_speed_scale * transform1.forward * z_axis_move;
            var x_vector = m_position_config.lateral_speed_scale * transform1.right * x_axis_move;
            var y_vector = transform1.up * -y_axis_rot;
            transform1.Rotate( y_vector, Time.deltaTime * Environment.player_base_angular_speed );
            Rigidbody.AddForce( (x_vector + z_vector) * Environment.player_base_speed, ForceMode.VelocityChange );
        }

    #endregion

    }
}
