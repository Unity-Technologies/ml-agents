//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Sounds for the bow pulling back
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SoundBowClick : MonoBehaviour
	{
		public AudioClip bowClick;
		public AnimationCurve pitchTensionCurve;
		public float minPitch;
		public float maxPitch;

		AudioSource thisAudioSource;

		//-------------------------------------------------
		void Awake()
		{
			thisAudioSource = GetComponent<AudioSource>();
		}


		//-------------------------------------------------
		public void PlayBowTensionClicks( float normalizedTension )
		{
			// Tension is a float between 0 and 1. 1 being max tension and 0 being no tension
			float y = pitchTensionCurve.Evaluate( normalizedTension );

			thisAudioSource.pitch = ( ( maxPitch - minPitch ) * y ) + minPitch;
			thisAudioSource.PlayOneShot( bowClick );
		}
	}
}
