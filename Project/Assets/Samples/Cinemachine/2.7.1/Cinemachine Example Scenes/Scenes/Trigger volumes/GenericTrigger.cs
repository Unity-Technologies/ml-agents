using UnityEngine;
using UnityEngine.Playables;

namespace Cinemachine.Examples
{

    public class GenericTrigger : MonoBehaviour
    {
        public PlayableDirector timeline;

        // Use this for initialization
        void Start()
        {
            timeline = GetComponent<PlayableDirector>();
        }

        void OnTriggerExit(Collider c)
        {
            if (c.gameObject.CompareTag("Player"))
            {
                // Jump to the end of the timeline where the blend happens
                // This value (in seconds) needs to be adjusted as needed if the timeline is modified
                timeline.time = 27;
            }
        }

        void OnTriggerEnter(Collider c)
        {
            if (c.gameObject.CompareTag("Player"))
            {
                timeline.Stop(); // Make sure the timeline is stopped before starting it
                timeline.Play();
            }
        }
    }

}
