using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(AudioSource))]
[RequireComponent(typeof(GunController))]

public class GunSFX : MonoBehaviour
{


    public AudioClip BackgroundSong;

    private AudioSource m_AudioSource;
    public Vector2 SoundScaleRandomRange = new Vector2(.2f, .4f);
    void Awake()
    {
        if (m_AudioSource == null)
        {
            m_AudioSource = GetComponent<AudioSource>();
            //            m_AudioSource = gameObject.AddComponent<AudioSource>();
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void PlayAudio()
    {
        m_AudioSource.PlayOneShot(m_AudioSource.clip);
    }
}
