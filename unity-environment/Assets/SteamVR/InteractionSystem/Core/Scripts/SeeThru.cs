//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Uses the see thru renderer while attached to hand
//
//=============================================================================

using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class SeeThru : MonoBehaviour
	{
		public Material seeThruMaterial;

		private GameObject seeThru;
		private Interactable interactable;
		private Renderer sourceRenderer;
		private Renderer destRenderer;


		//-------------------------------------------------
		void Awake()
		{
			interactable = GetComponentInParent<Interactable>();

			//
			// Create child game object for see thru renderer
			//
			seeThru = new GameObject( "_see_thru" );
			seeThru.transform.parent = transform;
			seeThru.transform.localPosition = Vector3.zero;
			seeThru.transform.localRotation = Quaternion.identity;
			seeThru.transform.localScale = Vector3.one;

			//
			// Copy mesh filter
			//
			MeshFilter sourceMeshFilter = GetComponent<MeshFilter>();
			if ( sourceMeshFilter != null )
			{
				MeshFilter destMeshFilter = seeThru.AddComponent<MeshFilter>();
				destMeshFilter.sharedMesh = sourceMeshFilter.sharedMesh;
			}

			//
			// Copy mesh renderer
			//
			MeshRenderer sourceMeshRenderer = GetComponent<MeshRenderer>();
			if ( sourceMeshRenderer != null )
			{
				sourceRenderer = sourceMeshRenderer;
				destRenderer = seeThru.AddComponent<MeshRenderer>();
			}

			//
			// Copy skinned mesh renderer
			//
			SkinnedMeshRenderer sourceSkinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
			if ( sourceSkinnedMeshRenderer != null )
			{
				SkinnedMeshRenderer destSkinnedMeshRenderer = seeThru.AddComponent<SkinnedMeshRenderer>();

				sourceRenderer = sourceSkinnedMeshRenderer;
				destRenderer = destSkinnedMeshRenderer;

				destSkinnedMeshRenderer.sharedMesh = sourceSkinnedMeshRenderer.sharedMesh;
				destSkinnedMeshRenderer.rootBone = sourceSkinnedMeshRenderer.rootBone;
				destSkinnedMeshRenderer.bones = sourceSkinnedMeshRenderer.bones;
				destSkinnedMeshRenderer.quality = sourceSkinnedMeshRenderer.quality;
				destSkinnedMeshRenderer.updateWhenOffscreen = sourceSkinnedMeshRenderer.updateWhenOffscreen;
			}

			//
			// Create see thru materials
			//
			if ( sourceRenderer != null && destRenderer != null )
			{
				int materialCount = sourceRenderer.sharedMaterials.Length;
				Material[] destRendererMaterials = new Material[materialCount];
				for ( int i = 0; i < materialCount; i++ )
				{
					destRendererMaterials[i] = seeThruMaterial;
				}
				destRenderer.sharedMaterials = destRendererMaterials;

				for ( int i = 0; i < destRenderer.materials.Length; i++ )
				{
					destRenderer.materials[i].renderQueue = 2001; // Rendered after geometry
				}

				for ( int i = 0; i < sourceRenderer.materials.Length; i++ )
				{
					if ( sourceRenderer.materials[i].renderQueue == 2000 )
					{
						sourceRenderer.materials[i].renderQueue = 2002;
					}
				}
			}

			seeThru.gameObject.SetActive( false );
		}


		//-------------------------------------------------
		void OnEnable()
		{
			interactable.onAttachedToHand += AttachedToHand;
			interactable.onDetachedFromHand += DetachedFromHand;
		}


		//-------------------------------------------------
		void OnDisable()
		{
			interactable.onAttachedToHand -= AttachedToHand;
			interactable.onDetachedFromHand -= DetachedFromHand;
		}


		//-------------------------------------------------
		private void AttachedToHand( Hand hand )
		{
			seeThru.SetActive( true );
		}


		//-------------------------------------------------
		private void DetachedFromHand( Hand hand )
		{
			seeThru.SetActive( false );
		}


		//-------------------------------------------------
		void Update()
		{
			if ( seeThru.activeInHierarchy )
			{
				int materialCount = Mathf.Min( sourceRenderer.materials.Length, destRenderer.materials.Length );
				for ( int i = 0; i < materialCount; i++ )
				{
					destRenderer.materials[i].mainTexture = sourceRenderer.materials[i].mainTexture;
					destRenderer.materials[i].color = destRenderer.materials[i].color * sourceRenderer.materials[i].color;
				}
			}
		}
	}
}
