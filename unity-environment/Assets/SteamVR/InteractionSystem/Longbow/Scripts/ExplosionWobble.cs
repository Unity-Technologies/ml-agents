//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Makes the weeble wobble
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class ExplosionWobble : MonoBehaviour
	{
		//-------------------------------------------------
		public void ExplosionEvent( Vector3 explosionPos )
		{
			var rb = GetComponent<Rigidbody>();
			if ( rb )
			{
				rb.AddExplosionForce( 2000, explosionPos, 10.0f );
			}
		}
	}
}
