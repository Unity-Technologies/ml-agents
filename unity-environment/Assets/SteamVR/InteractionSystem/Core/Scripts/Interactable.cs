//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: This object will get hover events and can be attached to the hands
//
//=============================================================================

using UnityEngine;
using UnityEngine.Events;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class Interactable : MonoBehaviour
	{
		public delegate void OnAttachedToHandDelegate( Hand hand );
		public delegate void OnDetachedFromHandDelegate( Hand hand );

		[HideInInspector]
		public event OnAttachedToHandDelegate onAttachedToHand;
		[HideInInspector]
		public event OnDetachedFromHandDelegate onDetachedFromHand;

		//-------------------------------------------------
		private void OnAttachedToHand( Hand hand )
		{
			if ( onAttachedToHand != null )
			{
				onAttachedToHand.Invoke( hand );
			}
		}


		//-------------------------------------------------
		private void OnDetachedFromHand( Hand hand )
		{
			if ( onDetachedFromHand != null )
			{
				onDetachedFromHand.Invoke( hand );
			}
		}
	}
}
