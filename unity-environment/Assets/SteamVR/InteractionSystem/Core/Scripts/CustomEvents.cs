//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Custom Unity Events that take in additional parameters
//
//=============================================================================

using UnityEngine.Events;
using System;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public static class CustomEvents
	{
		//-------------------------------------------------
		[System.Serializable]
		public class UnityEventSingleFloat : UnityEvent<float>
		{
		}


		//-------------------------------------------------
		[System.Serializable]
		public class UnityEventHand : UnityEvent<Hand>
		{
		}
	}
}
