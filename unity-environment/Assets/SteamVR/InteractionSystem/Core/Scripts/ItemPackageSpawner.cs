//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Handles the spawning and returning of the ItemPackage
//
//=============================================================================

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine.Events;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	[RequireComponent( typeof( Interactable ) )]
	public class ItemPackageSpawner : MonoBehaviour
	{
		public ItemPackage itemPackage
		{
			get
			{
				return _itemPackage;
			}
			set
			{
				CreatePreviewObject();
			}
		}

		public ItemPackage _itemPackage;

		private bool useItemPackagePreview = true;
		private bool useFadedPreview = false;
		private GameObject previewObject;

		public bool requireTriggerPressToTake = false;
		public bool requireTriggerPressToReturn = false;
		public bool showTriggerHint = false;

		[EnumFlags]
		public Hand.AttachmentFlags attachmentFlags = Hand.defaultAttachmentFlags;
		public string attachmentPoint;

		public bool takeBackItem = false; // if a hand enters this trigger and has the item this spawner dispenses at the top of the stack, remove it from the stack

		public bool acceptDifferentItems = false;

		private GameObject spawnedItem;
		private bool itemIsSpawned = false;

		public UnityEvent pickupEvent;
		public UnityEvent dropEvent;

		public bool justPickedUpItem = false;


		//-------------------------------------------------
		private void CreatePreviewObject()
		{
			if ( !useItemPackagePreview )
			{
				return;
			}

			ClearPreview();

			if ( useItemPackagePreview )
			{
				if ( itemPackage == null )
				{
					return;
				}

				if ( useFadedPreview == false ) // if we don't have a spawned item out there, use the regular preview
				{
					if ( itemPackage.previewPrefab != null )
					{
						previewObject = Instantiate( itemPackage.previewPrefab, transform.position, Quaternion.identity ) as GameObject;
						previewObject.transform.parent = transform;
						previewObject.transform.localRotation = Quaternion.identity;
					}
				}
				else // there's a spawned item out there. Use the faded preview
				{
					if ( itemPackage.fadedPreviewPrefab != null )
					{
						previewObject = Instantiate( itemPackage.fadedPreviewPrefab, transform.position, Quaternion.identity ) as GameObject;
						previewObject.transform.parent = transform;
						previewObject.transform.localRotation = Quaternion.identity;
					}
				}
			}
		}


		//-------------------------------------------------
		void Start()
		{
			VerifyItemPackage();
		}


		//-------------------------------------------------
		private void VerifyItemPackage()
		{
			if ( itemPackage == null )
			{
				ItemPackageNotValid();
			}

			if ( itemPackage.itemPrefab == null )
			{
				ItemPackageNotValid();
			}
		}


		//-------------------------------------------------
		private void ItemPackageNotValid()
		{
			Debug.LogError( "ItemPackage assigned to " + gameObject.name + " is not valid. Destroying this game object." );
			Destroy( gameObject );
		}


		//-------------------------------------------------
		private void ClearPreview()
		{
			foreach ( Transform child in transform )
			{
				if ( Time.time > 0 )
				{
					GameObject.Destroy( child.gameObject );
				}
				else
				{
					GameObject.DestroyImmediate( child.gameObject );
				}
			}
		}


		//-------------------------------------------------
		void Update()
		{
			if ( ( itemIsSpawned == true ) && ( spawnedItem == null ) )
			{
				itemIsSpawned = false;
				useFadedPreview = false;
				dropEvent.Invoke();
				CreatePreviewObject();
			}
		}


		//-------------------------------------------------
		private void OnHandHoverBegin( Hand hand )
		{
			ItemPackage currentAttachedItemPackage = GetAttachedItemPackage( hand );

			if ( currentAttachedItemPackage == itemPackage ) // the item at the top of the hand's stack has an associated ItemPackage
			{
				if ( takeBackItem && !requireTriggerPressToReturn ) // if we want to take back matching items and aren't waiting for a trigger press
				{
					TakeBackItem( hand );
				}
			}

			if ( !requireTriggerPressToTake ) // we don't require trigger press for pickup. Spawn and attach object.
			{
				SpawnAndAttachObject( hand );
			}

			if ( requireTriggerPressToTake && showTriggerHint )
			{
				ControllerButtonHints.ShowTextHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger, "PickUp" );
			}
		}


		//-------------------------------------------------
		private void TakeBackItem( Hand hand )
		{
			RemoveMatchingItemsFromHandStack( itemPackage, hand );

			if ( itemPackage.packageType == ItemPackage.ItemPackageType.TwoHanded )
			{
				RemoveMatchingItemsFromHandStack( itemPackage, hand.otherHand );
			}
		}


		//-------------------------------------------------
		private ItemPackage GetAttachedItemPackage( Hand hand )
		{
			GameObject currentAttachedObject = hand.currentAttachedObject;

			if ( currentAttachedObject == null ) // verify the hand is holding something
			{
				return null;
			}

			ItemPackageReference packageReference = hand.currentAttachedObject.GetComponent<ItemPackageReference>();
			if ( packageReference == null ) // verify the item in the hand is matchable
			{
				return null;
			}

			ItemPackage attachedItemPackage = packageReference.itemPackage; // return the ItemPackage reference we find.

			return attachedItemPackage;
		}


		//-------------------------------------------------
		private void HandHoverUpdate( Hand hand )
		{
			if ( requireTriggerPressToTake )
			{
				if ( hand.controller != null && hand.controller.GetHairTriggerDown() )
				{
					SpawnAndAttachObject( hand );
				}
			}
		}


		//-------------------------------------------------
		private void OnHandHoverEnd( Hand hand )
		{
			if ( !justPickedUpItem && requireTriggerPressToTake && showTriggerHint )
			{
				ControllerButtonHints.HideTextHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
			}

			justPickedUpItem = false;
		}


		//-------------------------------------------------
		private void RemoveMatchingItemsFromHandStack( ItemPackage package, Hand hand )
		{
			for ( int i = 0; i < hand.AttachedObjects.Count; i++ )
			{
				ItemPackageReference packageReference = hand.AttachedObjects[i].attachedObject.GetComponent<ItemPackageReference>();
				if ( packageReference != null )
				{
					ItemPackage attachedObjectItemPackage = packageReference.itemPackage;
					if ( ( attachedObjectItemPackage != null ) && ( attachedObjectItemPackage == package ) )
					{
						GameObject detachedItem = hand.AttachedObjects[i].attachedObject;
						hand.DetachObject( detachedItem );
					}
				}
			}
		}


		//-------------------------------------------------
		private void RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType packageType, Hand hand )
		{
			for ( int i = 0; i < hand.AttachedObjects.Count; i++ )
			{
				ItemPackageReference packageReference = hand.AttachedObjects[i].attachedObject.GetComponent<ItemPackageReference>();
				if ( packageReference != null )
				{
					if ( packageReference.itemPackage.packageType == packageType )
					{
						GameObject detachedItem = hand.AttachedObjects[i].attachedObject;
						hand.DetachObject( detachedItem );
					}
				}
			}
		}


		//-------------------------------------------------
		private void SpawnAndAttachObject( Hand hand )
		{
			if ( hand.otherHand != null )
			{
				//If the other hand has this item package, take it back from the other hand
				ItemPackage otherHandItemPackage = GetAttachedItemPackage( hand.otherHand );
				if ( otherHandItemPackage == itemPackage )
				{
					TakeBackItem( hand.otherHand );
				}
			}

			if ( showTriggerHint )
			{
				ControllerButtonHints.HideTextHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
			}

			if ( itemPackage.otherHandItemPrefab != null )
			{
				if ( hand.otherHand.hoverLocked )
				{
					//Debug.Log( "Not attaching objects because other hand is hoverlocked and we can't deliver both items." );
					return;
				}
			}

			// if we're trying to spawn a one-handed item, remove one and two-handed items from this hand and two-handed items from both hands
			if ( itemPackage.packageType == ItemPackage.ItemPackageType.OneHanded )
			{
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.OneHanded, hand );
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.TwoHanded, hand );
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.TwoHanded, hand.otherHand );
			}

			// if we're trying to spawn a two-handed item, remove one and two-handed items from both hands
			if ( itemPackage.packageType == ItemPackage.ItemPackageType.TwoHanded )
			{
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.OneHanded, hand );
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.OneHanded, hand.otherHand );
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.TwoHanded, hand );
				RemoveMatchingItemTypesFromHand( ItemPackage.ItemPackageType.TwoHanded, hand.otherHand );
			}

			spawnedItem = GameObject.Instantiate( itemPackage.itemPrefab );
			spawnedItem.SetActive( true );
			hand.AttachObject( spawnedItem, attachmentFlags, attachmentPoint );

			if ( ( itemPackage.otherHandItemPrefab != null ) && ( hand.otherHand.controller != null ) )
			{
				GameObject otherHandObjectToAttach = GameObject.Instantiate( itemPackage.otherHandItemPrefab );
				otherHandObjectToAttach.SetActive( true );
				hand.otherHand.AttachObject( otherHandObjectToAttach, attachmentFlags );
			}

			itemIsSpawned = true;

			justPickedUpItem = true;

			if ( takeBackItem )
			{
				useFadedPreview = true;
				pickupEvent.Invoke();
				CreatePreviewObject();
			}
		}
	}
}
