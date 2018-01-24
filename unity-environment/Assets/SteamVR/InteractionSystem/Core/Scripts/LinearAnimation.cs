//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Animation that moves based on a linear mapping
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class LinearAnimation : MonoBehaviour
	{
		public LinearMapping linearMapping;
		public new Animation animation;

		private AnimationState animState;
		private float animLength;
		private float lastValue;

	
		//-------------------------------------------------
		void Awake()
		{
			if ( animation == null )
			{
				animation = GetComponent<Animation>();
			}

			if ( linearMapping == null )
			{
				linearMapping = GetComponent<LinearMapping>();
			}

			//We're assuming the animation has a single clip, and that's the one we're
			//going to scrub with the linear mapping.
			animation.playAutomatically = true;
			animState = animation[animation.clip.name];

			//If the anim state's (i.e. clip's) wrap mode is Once (the default) or ClampForever,
			//Unity will automatically stop playing the anim, regardless of subsequent changes
			//to animState.time. Thus, we set the wrap mode to PingPong.
			animState.wrapMode = WrapMode.PingPong;
			animState.speed = 0;
			animLength = animState.length;
		}


		//-------------------------------------------------
		void Update()
		{
			float value = linearMapping.value;

			//No need to set the anim if our value hasn't changed.
			if ( value != lastValue )
			{
				animState.time = value / animLength;
			}

			lastValue = value;
		}
	}
}
