using UnityEngine;
namespace Project
{
    public class MSoccerBall : MonoBehaviour
    {

    #region Config

        public string GOAL_TAG;
        string[] goal_tags;
        public string PLAYER_TAG;
        string[] player_tags;
        Vector3 start_position;

    #endregion

    #region Reference

        MSoccerEnvironment Environment;
        Rigidbody Rigidbody;

    #endregion

    #region State

        MSoccerPlayerAgent last_agent;

    #endregion



        void Start()
        {
            goal_tags = new string[2];
            for (int i = 0; i < goal_tags.Length; i++)
            {
                goal_tags[i] = $"{GOAL_TAG}_{i}";
            }
            player_tags = new string[2];
            for (int i = 0; i < player_tags.Length; i++)
            {
                player_tags[i] = $"{PLAYER_TAG}_{i}";
            }
            Environment = GetComponentInParent<MSoccerEnvironment>();
            Rigidbody = GetComponent<Rigidbody>();
            start_position = transform.position;
        }

        void OnCollisionEnter(Collision collision)
        {
            var collided = collision.gameObject;
            if (collided.CompareTag( goal_tags[0] ))
            {
                Environment.Goal( 0, last_agent );
            }
            if (collided.CompareTag( goal_tags[1] ))
            {
                Environment.Goal( 1, last_agent );
            }
            if (collided.CompareTag( player_tags[0] )||collided.CompareTag( player_tags[1] ))
            {
                last_agent= collided.GetComponent<MSoccerPlayerAgent>();
            }
            // if (collided.CompareTag( "wall" ))
            // {
            //     Environment.Goal( last_kick_team_id_reverse );
            // }


        }

        public void Init()
        {
            transform.position = start_position + new Vector3(
                Random.Range( -2.5f, 2.5f ), 0,
                Random.Range( -2.5f, 2.5f ) );
            Rigidbody.velocity = Vector3.zero;
            Rigidbody.angularVelocity = Vector3.zero;
        }
    }
}
