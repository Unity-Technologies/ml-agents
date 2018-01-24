//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Spawns a render model for the controller from SteamVR
//
//=============================================================================

using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SpawnRenderModel : MonoBehaviour
	{
		public Material[] materials;

		private SteamVR_RenderModel[] renderModels;
		private Hand hand;
		private List<MeshRenderer> renderers = new List<MeshRenderer>();

		private static List<SpawnRenderModel> spawnRenderModels = new List<SpawnRenderModel>();
		private static int lastFrameUpdated;
		private static int spawnRenderModelUpdateIndex;

		SteamVR_Events.Action renderModelLoadedAction;


		//-------------------------------------------------
		void Awake()
		{
			renderModels = new SteamVR_RenderModel[materials.Length];
			renderModelLoadedAction = SteamVR_Events.RenderModelLoadedAction( OnRenderModelLoaded );
		}


		//-------------------------------------------------
		void OnEnable()
		{
			ShowController();

			renderModelLoadedAction.enabled = true;

			spawnRenderModels.Add( this );
		}


		//-------------------------------------------------
		void OnDisable()
		{
			HideController();

			renderModelLoadedAction.enabled = false;

			spawnRenderModels.Remove( this );
		}


		//-------------------------------------------------
		private void OnAttachedToHand( Hand hand )
		{
			this.hand = hand;
			ShowController();
		}


		//-------------------------------------------------
		private void OnDetachedFromHand( Hand hand )
		{
			this.hand = null;
			HideController();
		}


		//-------------------------------------------------
		void Update()
		{
			// Only update one per frame
			if ( lastFrameUpdated == Time.renderedFrameCount )
			{
				return;
			}
			lastFrameUpdated = Time.renderedFrameCount;


			// SpawnRenderModel overflow
			if ( spawnRenderModelUpdateIndex >= spawnRenderModels.Count )
			{
				spawnRenderModelUpdateIndex = 0;
			}


			// Perform update
			if ( spawnRenderModelUpdateIndex < spawnRenderModels.Count )
			{
				SteamVR_RenderModel renderModel = spawnRenderModels[spawnRenderModelUpdateIndex].renderModels[0];
				if ( renderModel != null )
				{
					renderModel.UpdateComponents( OpenVR.RenderModels );
				}
			}

			spawnRenderModelUpdateIndex++;
		}


		//-------------------------------------------------
		private void ShowController()
		{
			if ( hand == null || hand.controller == null )
			{
				return;
			}

			for ( int i = 0; i < renderModels.Length; i++ )
			{
				if ( renderModels[i] == null )
				{
					renderModels[i] = new GameObject( "SteamVR_RenderModel" ).AddComponent<SteamVR_RenderModel>();
					renderModels[i].updateDynamically = false; // Update one per frame (see Update() method)
					renderModels[i].transform.parent = transform;
					Util.ResetTransform( renderModels[i].transform );
				}

				renderModels[i].gameObject.SetActive( true );
				renderModels[i].SetDeviceIndex( (int)hand.controller.index );
			}
		}


		//-------------------------------------------------
		private void HideController()
		{
			for ( int i = 0; i < renderModels.Length; i++ )
			{
				if ( renderModels[i] != null )
				{
					renderModels[i].gameObject.SetActive( false );
				}
			}
		}


		//-------------------------------------------------
		private void OnRenderModelLoaded( SteamVR_RenderModel renderModel, bool success )
		{
			for ( int i = 0; i < renderModels.Length; i++ )
			{
				if ( renderModel == renderModels[i] )
				{
					if ( materials[i] != null )
					{
						renderers.Clear();
						renderModels[i].GetComponentsInChildren<MeshRenderer>( renderers );
						for ( int j = 0; j < renderers.Count; j++ )
						{
							Texture mainTexture = renderers[j].material.mainTexture;
							renderers[j].sharedMaterial = materials[i];
							renderers[j].material.mainTexture = mainTexture;
							renderers[j].gameObject.layer = gameObject.layer;
							renderers[j].tag = gameObject.tag;
						}
					}
				}
			}
		}
	}
}
