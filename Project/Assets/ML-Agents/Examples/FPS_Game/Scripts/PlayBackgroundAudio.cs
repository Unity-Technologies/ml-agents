using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;
using UnityEngine;


//songs used
// https://www.youtube.com/watch?v=gXm1W2eJ0e0
[RequireComponent(typeof(AudioSource))]
public class PlayBackgroundAudio : MonoBehaviour
{

    //    public AudioClip BackgroundSong;

    private AudioSource m_AudioSource;
    void Awake()
    {
        //        DontDestroyOnLoad(gameObject);

        if (m_AudioSource == null)
        {
            m_AudioSource = GetComponent<AudioSource>();
            //            m_AudioSource = gameObject.AddComponent<AudioSource>();
        }
    }
    // Start is called before the first frame update

    void Start()
    {
        //        if (BackgroundSong)
        //        {
        //            m_AudioSource.clip = BackgroundSong;
        if (!m_AudioSource.isPlaying)
        {
            m_AudioSource.Play();
        }
        //        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
