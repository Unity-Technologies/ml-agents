//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Stores the play area size info from the players chaperone data
//
//=============================================================================

using UnityEngine;
using UnityEngine.Events;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class ChaperoneInfo : MonoBehaviour
	{
		public bool initialized { get; private set; }
		public float playAreaSizeX { get; private set; }
		public float playAreaSizeZ { get; private set; }
		public bool roomscale { get; private set; }

		public static SteamVR_Events.Event Initialized = new SteamVR_Events.Event();
		public static SteamVR_Events.Action InitializedAction( UnityAction action ) { return new SteamVR_Events.ActionNoArgs( Initialized, action ); }

		//-------------------------------------------------
		private static ChaperoneInfo _instance;
		public static ChaperoneInfo instance
		{
			get
			{
				if ( _instance == null )
				{
					_instance = new GameObject( "[ChaperoneInfo]" ).AddComponent<ChaperoneInfo>();
					_instance.initialized = false;
					_instance.playAreaSizeX = 1.0f;
					_instance.playAreaSizeZ = 1.0f;
					_instance.roomscale = false;

					DontDestroyOnLoad( _instance.gameObject );
				}
				return _instance;
			}
		}


		//-------------------------------------------------
		IEnumerator Start()
		{
			// Uncomment for roomscale testing
			//_instance.initialized = true;
			//_instance.playAreaSizeX = UnityEngine.Random.Range( 1.0f, 4.0f );
			//_instance.playAreaSizeZ = UnityEngine.Random.Range( 1.0f, _instance.playAreaSizeX );
			//_instance.roomscale = true;
			//ChaperoneInfo.Initialized.Send();
			//yield break;

			// Get interface pointer
			var chaperone = OpenVR.Chaperone;
			if ( chaperone == null )
			{
				Debug.LogWarning( "Failed to get IVRChaperone interface." );
				initialized = true;
				yield break;
			}

			// Get play area size
			while ( true )
			{
				float px = 0.0f, pz = 0.0f;
				if ( chaperone.GetPlayAreaSize( ref px, ref pz ) )
				{
					initialized = true;
					playAreaSizeX = px;
					playAreaSizeZ = pz;
					roomscale = Mathf.Max( px, pz ) > 1.01f;

					Debug.LogFormat( "ChaperoneInfo initialized. {2} play area {0:0.00}m x {1:0.00}m", px, pz, roomscale ? "Roomscale" : "Standing" );

					ChaperoneInfo.Initialized.Send();

					yield break;
				}

				yield return null;
			}
		}
	}
}
