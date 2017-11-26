//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Debug UI shown for the player
//
//=============================================================================

using UnityEngine;
using UnityEngine.UI;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class DebugUI : MonoBehaviour
	{
		private Player player;

		//-------------------------------------------------
		static private DebugUI _instance;
		static public DebugUI instance
		{
			get
			{
				if ( _instance == null )
				{
					_instance = GameObject.FindObjectOfType<DebugUI>();
				}
				return _instance;
			}
		}


		//-------------------------------------------------
		void Start()
		{
			player = Player.instance;
		}


		//-------------------------------------------------
		private void OnGUI()
		{
#if !HIDE_DEBUG_UI
			player.Draw2DDebug();
#endif
		}
	}
}
