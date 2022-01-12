using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
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
        int team_id_reverse;
        public int position_id;
        PositionConfig m_position_config;
        Vector3 spawn_position;
        Quaternion spawn_rotation;

        public bool freeze = false;

    #endregion

    #region State

        int m_action_kick_power;

    #endregion

    #region AgentEvent

        public override void Initialize()
        {
            Parameters = GetComponent<BehaviorParameters>();
            Environment = GetComponentInParent<MSoccerEnvironment>();
            team_id = Parameters.TeamId;
            team_id_reverse = team_id == 0 ? 1 : 0;
            Rigidbody = GetComponent<Rigidbody>();
            Rigidbody.maxAngularVelocity = 500;
            m_position_config = Environment.positions[position_id];
            var transform1 = transform;
            spawn_position = transform1.position;
            spawn_rotation = transform1.rotation;

        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation( Environment.ball_transform.position - transform.position );
            sensor.AddObservation( Environment.ball_rigidbody.velocity - Rigidbody.velocity );
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
                m_action_kick_power * Environment.player_kick_power * m_position_config.kick_power_scale *
                Mathf.Clamp( Vector3.Dot( Rigidbody.velocity, transform.forward ), 0, 10 );
            var direction = (collided.transform.position - transform.position).normalized;
            collision.rigidbody.AddForce( direction * force );

        }
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // var acts = actionsOut.DiscreteActions;
            // //forward
            // if (Input.GetKey( KeyCode.W ))
            // {
            //     acts[0] = 1;
            // }
            // if (Input.GetKey( KeyCode.S ))
            // {
            //     acts[0] = 2;
            // }
            //
            // //rotate
            // if (Input.GetKey( KeyCode.E ))
            // {
            //     acts[2] = 1;
            // }
            // if (Input.GetKey( KeyCode.Q ))
            // {
            //     acts[2] = 2;
            // }
            //
            // //right
            // if (Input.GetKey( KeyCode.D ))
            // {
            //     acts[1] = 1;
            // }
            // if (Input.GetKey( KeyCode.A ))
            // {
            //     acts[1] = 2;
            // }

            HigherHeuristic( in actionsOut );


        }

        // public override void CollectObservations(VectorSensor sensor) { }
        public override void OnActionReceived(ActionBuffers actionsBuffers)
        {
            if (freeze)
            {
                return;
            }

            AddReward( m_position_config.timer_reward *
                       Environment.step_ratio );

            var position = Environment.ball_transform.position;
            var position1 = transform.position;
            AddReward( 0.01f * Mathf.Exp( -0.1f *
                                          Vector3.Distance(
                                              position,
                                              position1 ) ) );
            AddReward( Vector3.Distance(
                position,
                position1 ) < 0.1f ?
                0.01f / Vector3.Distance( Environment.ball_rigidbody.velocity, Rigidbody.velocity ) : 0 );

            AddReward( 0.01f / (1 +
                                Vector3.Distance(
                                    Environment.goal_transforms[team_id_reverse].position,
                                    position )) );

            var acts = actionsBuffers.DiscreteActions;

            m_action_kick_power = acts[3];

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

            AddReward( -Environment.step_ratio * 2 * Mathf.Exp( -2f / (0.01f + Rigidbody.velocity.magnitude) ) );
            AddReward( -Environment.step_ratio * 4 * Mathf.Exp( -2f / (0.01f + Rigidbody.angularVelocity.magnitude) ) );
        }



        Vector3 Destination;
        public void HigherHeuristic(in ActionBuffers actionsOut)
        {
            var acts = actionsOut.DiscreteActions;
            if (Input.GetMouseButton( 0 ))
            {
                if (Physics.Raycast( Camera.main.ScreenPointToRay( Input.mousePosition ), out var hitInfo, 1000000f, LayerMask.GetMask( "Ground" ) ))
                {
                    Destination = hitInfo.point;
                }
            }

            Vector3 vector = Destination - transform.position;
            Vector3 dir = vector.normalized;

            //left?
            if (Vector3.Cross( transform.forward, dir ).y > 0.1f)
            {
                acts[2] = 2;
            }
            else if (Vector3.Cross( transform.forward, dir ).y < -0.1f)
            {
                acts[2] = 1;
            }
            else
            {
                acts[2] = 0;
            }

            if (Vector3.Dot( transform.forward, dir ) > 0.9f)
            {
                acts[0] = 1;
            }
            else if (Vector3.Dot( transform.forward, dir ) < -1f)
            {
                acts[0] = 2;
            }
            else
            {
                acts[0] = 0;
            }

            if (vector.magnitude < 0.1f)
            {
                acts[0] = 0;
            }


            if (Input.GetKey( KeyCode.A ))
            {
                acts[1] = 1;
            }
            else if (Input.GetKey( KeyCode.D ))
            {
                acts[1] = 2;
            }
            else
            {
                acts[1] = 0;
            }


            if (Input.GetKey( KeyCode.Space ))
            {
                acts[3] = 1;
            }
            else
            {
                acts[3] = 0;
            }
        }

    #endregion

    }
}
