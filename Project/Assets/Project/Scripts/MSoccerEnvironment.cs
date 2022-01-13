using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using UnityEngine;
namespace Project
{

    [Serializable]
    public struct PositionConfig
    {
        public string position_name;
        public float forward_speed_scale;
        public float lateral_speed_scale;
        public float kick_power_scale;
        public float timer_reward;
    }

    public class MSoccerEnvironment : MonoBehaviour
    {

    #region Reference

        EnvironmentParameters environmentParameters;
        public Transform ball_transform;
        public Rigidbody ball_rigidbody;
        public Transform[] goal_transforms;

    #endregion

    #region Config

        /// <summary>
        ///     One FixedUpdate -> One step. Reaching max step causes episode reset.
        /// </summary>
        public int episode_max_steps = 25000;
        public float player_base_speed = 2;
        public float player_base_angular_speed = 100;
        public float player_kick_power = 200;
        public float player_touch_ball_reward = 0.2f;
        public float player_time_reward = 2f;
        public List<PositionConfig> positions;

    #endregion

    #region States

        List<MSoccerPlayerAgent> agents;
        SimpleMultiAgentGroup[] agent_groups;
        MSoccerBall soccer_ball;
        /// <summary>
        ///     The step timer of current episode.
        /// </summary>
        public int cur_step { private set; get; }
        public float step_ratio => 1f / episode_max_steps;

    #endregion

    #region UnityEvents

        void Start()
        {
            InitEnvironment();
        }

        void FixedUpdate()
        {
            if (cur_step >= episode_max_steps)
            {
                EndEpisode( true );
                InitEpisode();
            }
            cur_step++;
        }

    #endregion

    #region Actions

    #region Initiation

        void InitEnvironment()
        {
            environmentParameters = Academy.Instance.EnvironmentParameters;
            agents = GetComponentsInChildren<MSoccerPlayerAgent>().ToList();
            agent_groups = new SimpleMultiAgentGroup[2];
            agent_groups[0] = new SimpleMultiAgentGroup();
            agent_groups[1] = new SimpleMultiAgentGroup();
            foreach (var agent in agents)
            {
                agent_groups[agent.team_id].RegisterAgent( agent );
            }
            soccer_ball = GetComponentInChildren<MSoccerBall>();
            ball_transform = soccer_ball.transform;
            ball_rigidbody = soccer_ball.GetComponent<Rigidbody>();
            goal_transforms = new Transform[2];
            goal_transforms[0] = transform.Find( "Field" ).Find( "Goal_0" );
            goal_transforms[1] = transform.Find( "Field" ).Find( "Goal_1" );
        }

        void InitEpisode()
        {
            cur_step = 0;
            soccer_ball.Init();
            player_touch_ball_reward = environmentParameters.GetWithDefault( "ball_touch", player_touch_ball_reward );
        }

    #endregion

        void EndEpisode(bool interrupt)
        {
            if (interrupt)
            {
                agent_groups[0].GroupEpisodeInterrupted();
                agent_groups[1].GroupEpisodeInterrupted();
            }
            else
            {
                agent_groups[0].EndGroupEpisode();
                agent_groups[1].EndGroupEpisode();
            }

        }

    #region Interface

        public void Goal(int scored_team_id, MSoccerPlayerAgent agent)
        {
            int enemy_team_id = scored_team_id == 0 ? 1 : 0;
            float reward = 15 * (1 - cur_step * step_ratio);
            agent_groups[scored_team_id].AddGroupReward( reward );
            agent_groups[enemy_team_id].AddGroupReward( -7.5f );
            agent.AddReward( reward );
            EndEpisode( false );
            InitEpisode();
        }

    #endregion

    #endregion

    }
}
