//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: This object's rigidbody goes to sleep when created
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SleepOnAwake : MonoBehaviour
	{
		//-------------------------------------------------
		void Awake()
		{
			Rigidbody rigidbody = GetComponent<Rigidbody>();
			if ( rigidbody )
			{
				rigidbody.Sleep();
			}
		}
	}
}
