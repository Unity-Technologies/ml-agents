//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Spawns and attaches an object to the hand after the controller has
//			tracking
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SpawnAndAttachAfterControllerIsTracking : MonoBehaviour
	{
		private Hand hand;
		public GameObject itemPrefab;

	
		//-------------------------------------------------
		void Start()
		{
			hand = GetComponentInParent<Hand>();
		}


		//-------------------------------------------------
		void Update()
		{
			if ( itemPrefab != null )
			{
				if ( hand.controller != null )
				{
					if ( hand.controller.hasTracking )
					{
						GameObject objectToAttach = GameObject.Instantiate( itemPrefab );
						objectToAttach.SetActive( true );
						hand.AttachObject( objectToAttach );
						hand.controller.TriggerHapticPulse( 800 );
						Destroy( gameObject );

						// If the player's scale has been changed the object to attach will be the wrong size.
						// To fix this we change the object's scale back to its original, pre-attach scale.
						objectToAttach.transform.localScale = itemPrefab.transform.localScale;
					}
				}
			}
		}
	}
}
