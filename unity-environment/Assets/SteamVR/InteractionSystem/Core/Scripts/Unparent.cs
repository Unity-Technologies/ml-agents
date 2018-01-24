//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Unparents an object and keeps track of the old parent
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class Unparent : MonoBehaviour
	{
		Transform oldParent;

		//-------------------------------------------------
		void Start()
		{
			oldParent = transform.parent;
			transform.parent = null;
			gameObject.name = oldParent.gameObject.name + "." + gameObject.name;
		}


		//-------------------------------------------------
		void Update()
		{
			if ( oldParent == null )
				Object.Destroy( gameObject );
		}


		//-------------------------------------------------
		public Transform GetOldParent()
		{
			return oldParent;
		}
	}
}
