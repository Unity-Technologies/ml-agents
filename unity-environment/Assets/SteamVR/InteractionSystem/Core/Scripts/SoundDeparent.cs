//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Unparents this object and optionally destroys it after the sound
//			has played
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SoundDeparent : MonoBehaviour
	{
		public bool destroyAfterPlayOnce = true;
		private AudioSource thisAudioSource;


		//-------------------------------------------------
		void Awake()
		{
			thisAudioSource = GetComponent<AudioSource>();

		}


		//-------------------------------------------------
		void Start()
		{
			// move the sound object out from under the parent 
			gameObject.transform.parent = null;

			if ( destroyAfterPlayOnce )
				Destroy( gameObject, thisAudioSource.clip.length );
		}
	}
}
