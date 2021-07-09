using System.Collections;
using System.Collections.Generic;
using UnityEngine;


//songs used
//http://en.shw.in/
//http://shw.in/sozai/audio120402/tsudzumi-japan3.mp3
[RequireComponent(typeof(AudioSource))]
public class PlayBackgroundAudio : MonoBehaviour
{
    private AudioSource m_AudioSource;
    void Awake()
    {
        if (m_AudioSource == null)
        {
            m_AudioSource = GetComponent<AudioSource>();
        }
    }

    void Start()
    {
        if (!m_AudioSource.isPlaying)
        {
            m_AudioSource.Play();
        }
    }
}
