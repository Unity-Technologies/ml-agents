//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: The hands used by the player in the vr interaction system
//
//=============================================================================

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	// Links with an appropriate SteamVR controller and facilitates
	// interactions with objects in the virtual world.
	//-------------------------------------------------------------------------
	public class Hand : MonoBehaviour
	{
		public enum HandType
		{
			Left,
			Right,
			Any
		};

		// The flags used to determine how an object is attached to the hand.
		[Flags]
		public enum AttachmentFlags
		{
			SnapOnAttach = 1 << 0, // The object should snap to the position of the specified attachment point on the hand.
			DetachOthers = 1 << 1, // Other objects attached to this hand will be detached.
			DetachFromOtherHand = 1 << 2, // This object will be detached from the other hand.
			ParentToHand = 1 << 3, // The object will be parented to the hand.
		};

		public const AttachmentFlags defaultAttachmentFlags = AttachmentFlags.ParentToHand |
															  AttachmentFlags.DetachOthers |
															  AttachmentFlags.DetachFromOtherHand |
															  AttachmentFlags.SnapOnAttach;

		public Hand otherHand;
		public HandType startingHandType;

		public Transform hoverSphereTransform;
		public float hoverSphereRadius = 0.05f;
		public LayerMask hoverLayerMask = -1;
		public float hoverUpdateInterval = 0.1f;

		public Camera noSteamVRFallbackCamera;
		public float noSteamVRFallbackMaxDistanceNoItem = 10.0f;
		public float noSteamVRFallbackMaxDistanceWithItem = 0.5f;
		private float noSteamVRFallbackInteractorDistance = -1.0f;

		public SteamVR_Controller.Device controller;

		public GameObject controllerPrefab;
		private GameObject controllerObject = null;

		public bool showDebugText = false;
		public bool spewDebugText = false;

		public struct AttachedObject
		{
			public GameObject attachedObject;
			public GameObject originalParent;
			public bool isParentedToHand;
		}

		private List<AttachedObject> attachedObjects = new List<AttachedObject>();

		public ReadOnlyCollection<AttachedObject> AttachedObjects
		{
			get { return attachedObjects.AsReadOnly(); }
		}

		public bool hoverLocked { get; private set; }

		private Interactable _hoveringInteractable;

		private TextMesh debugText;
		private int prevOverlappingColliders = 0;

		private const int ColliderArraySize = 16;
		private Collider[] overlappingColliders;

		private Player playerInstance;

		private GameObject applicationLostFocusObject;

		SteamVR_Events.Action inputFocusAction;


		//-------------------------------------------------
		// The Interactable object this Hand is currently hovering over
		//-------------------------------------------------
		public Interactable hoveringInteractable
		{
			get { return _hoveringInteractable; }
			set
			{
				if ( _hoveringInteractable != value )
				{
					if ( _hoveringInteractable != null )
					{
						HandDebugLog( "HoverEnd " + _hoveringInteractable.gameObject );
						_hoveringInteractable.SendMessage( "OnHandHoverEnd", this, SendMessageOptions.DontRequireReceiver );

						//Note: The _hoveringInteractable can change after sending the OnHandHoverEnd message so we need to check it again before broadcasting this message
						if ( _hoveringInteractable != null )
						{
							this.BroadcastMessage( "OnParentHandHoverEnd", _hoveringInteractable, SendMessageOptions.DontRequireReceiver ); // let objects attached to the hand know that a hover has ended
						}
					}

					_hoveringInteractable = value;

					if ( _hoveringInteractable != null )
					{
						HandDebugLog( "HoverBegin " + _hoveringInteractable.gameObject );
						_hoveringInteractable.SendMessage( "OnHandHoverBegin", this, SendMessageOptions.DontRequireReceiver );

						//Note: The _hoveringInteractable can change after sending the OnHandHoverBegin message so we need to check it again before broadcasting this message
						if ( _hoveringInteractable != null )
						{
							this.BroadcastMessage( "OnParentHandHoverBegin", _hoveringInteractable, SendMessageOptions.DontRequireReceiver ); // let objects attached to the hand know that a hover has begun
						}
					}
				}
			}
		}


		//-------------------------------------------------
		// Active GameObject attached to this Hand
		//-------------------------------------------------
		public GameObject currentAttachedObject
		{
			get
			{
				CleanUpAttachedObjectStack();

				if ( attachedObjects.Count > 0 )
				{
					return attachedObjects[attachedObjects.Count - 1].attachedObject;
				}

				return null;
			}
		}


		//-------------------------------------------------
		public Transform GetAttachmentTransform( string attachmentPoint = "" )
		{
			Transform attachmentTransform = null;

			if ( !string.IsNullOrEmpty( attachmentPoint ) )
			{
				attachmentTransform = transform.Find( attachmentPoint );
			}

			if ( !attachmentTransform )
			{
				attachmentTransform = this.transform;
			}

			return attachmentTransform;
		}


		//-------------------------------------------------
		// Guess the type of this Hand
		//
		// If startingHandType is Hand.Left or Hand.Right, returns startingHandType.
		// If otherHand is non-null and both Hands are linked to controllers, returns
		// Hand.Left if this Hand is leftmost relative to the HMD, otherwise Hand.Right.
		// Otherwise, returns Hand.Any
		//-------------------------------------------------
		public HandType GuessCurrentHandType()
		{
			if ( startingHandType == HandType.Left || startingHandType == HandType.Right )
			{
				return startingHandType;
			}

			if ( startingHandType == HandType.Any && otherHand != null && otherHand.controller == null )
			{
				return HandType.Right;
			}

			if ( controller == null || otherHand == null || otherHand.controller == null )
			{
				return startingHandType;
			}

			if ( controller.index == SteamVR_Controller.GetDeviceIndex( SteamVR_Controller.DeviceRelation.Leftmost ) )
			{
				return HandType.Left;
			}

			return HandType.Right;
		}


		//-------------------------------------------------
		// Attach a GameObject to this GameObject
		//
		// objectToAttach - The GameObject to attach
		// flags - The flags to use for attaching the object
		// attachmentPoint - Name of the GameObject in the hierarchy of this Hand which should act as the attachment point for this GameObject
		//-------------------------------------------------
		public void AttachObject( GameObject objectToAttach, AttachmentFlags flags = defaultAttachmentFlags, string attachmentPoint = "" )
		{
			if ( flags == 0 )
			{
				flags = defaultAttachmentFlags;
			}

			//Make sure top object on stack is non-null
			CleanUpAttachedObjectStack();

			//Detach the object if it is already attached so that it can get re-attached at the top of the stack
			DetachObject( objectToAttach );

			//Detach from the other hand if requested
			if ( ( ( flags & AttachmentFlags.DetachFromOtherHand ) == AttachmentFlags.DetachFromOtherHand ) && otherHand )
			{
				otherHand.DetachObject( objectToAttach );
			}

			if ( ( flags & AttachmentFlags.DetachOthers ) == AttachmentFlags.DetachOthers )
			{
				//Detach all the objects from the stack
				while ( attachedObjects.Count > 0 )
				{
					DetachObject( attachedObjects[0].attachedObject );
				}
			}

			if ( currentAttachedObject )
			{
				currentAttachedObject.SendMessage( "OnHandFocusLost", this, SendMessageOptions.DontRequireReceiver );
			}

			AttachedObject attachedObject = new AttachedObject();
			attachedObject.attachedObject = objectToAttach;
			attachedObject.originalParent = objectToAttach.transform.parent != null ? objectToAttach.transform.parent.gameObject : null;
			if ( ( flags & AttachmentFlags.ParentToHand ) == AttachmentFlags.ParentToHand )
			{
				//Parent the object to the hand
				objectToAttach.transform.parent = GetAttachmentTransform( attachmentPoint );
				attachedObject.isParentedToHand = true;
			}
			else
			{
				attachedObject.isParentedToHand = false;
			}
			attachedObjects.Add( attachedObject );

			if ( ( flags & AttachmentFlags.SnapOnAttach ) == AttachmentFlags.SnapOnAttach )
			{
				objectToAttach.transform.localPosition = Vector3.zero;
				objectToAttach.transform.localRotation = Quaternion.identity;
			}

			HandDebugLog( "AttachObject " + objectToAttach );
			objectToAttach.SendMessage( "OnAttachedToHand", this, SendMessageOptions.DontRequireReceiver );

			UpdateHovering();
		}


		//-------------------------------------------------
		// Detach this GameObject from the attached object stack of this Hand
		//
		// objectToDetach - The GameObject to detach from this Hand
		//-------------------------------------------------
		public void DetachObject( GameObject objectToDetach, bool restoreOriginalParent = true )
		{
			int index = attachedObjects.FindIndex( l => l.attachedObject == objectToDetach );
			if ( index != -1 )
			{
				HandDebugLog( "DetachObject " + objectToDetach );

				GameObject prevTopObject = currentAttachedObject;

				Transform parentTransform = null;
				if ( attachedObjects[index].isParentedToHand )
				{
					if ( restoreOriginalParent && ( attachedObjects[index].originalParent != null ) )
					{
						parentTransform = attachedObjects[index].originalParent.transform;
					}
					attachedObjects[index].attachedObject.transform.parent = parentTransform;
				}

				attachedObjects[index].attachedObject.SetActive( true );
				attachedObjects[index].attachedObject.SendMessage( "OnDetachedFromHand", this, SendMessageOptions.DontRequireReceiver );
				attachedObjects.RemoveAt( index );

				GameObject newTopObject = currentAttachedObject;

				//Give focus to the top most object on the stack if it changed
				if ( newTopObject != null && newTopObject != prevTopObject )
				{
					newTopObject.SetActive( true );
					newTopObject.SendMessage( "OnHandFocusAcquired", this, SendMessageOptions.DontRequireReceiver );
				}
			}

			CleanUpAttachedObjectStack();
		}


		//-------------------------------------------------
		// Get the world velocity of the VR Hand.
		// Note: controller velocity value only updates on controller events (Button but and down) so good for throwing
		//-------------------------------------------------
		public Vector3 GetTrackedObjectVelocity()
		{
			if ( controller != null )
			{
				return transform.parent.TransformVector( controller.velocity );
			}

			return Vector3.zero;
		}


		//-------------------------------------------------
		// Get the world angular velocity of the VR Hand.
		// Note: controller velocity value only updates on controller events (Button but and down) so good for throwing
		//-------------------------------------------------
		public Vector3 GetTrackedObjectAngularVelocity()
		{
			if ( controller != null )
			{
				return transform.parent.TransformVector( controller.angularVelocity );
			}

			return Vector3.zero;
		}


		//-------------------------------------------------
		private void CleanUpAttachedObjectStack()
		{
			attachedObjects.RemoveAll( l => l.attachedObject == null );
		}


		//-------------------------------------------------
		void Awake()
		{
			inputFocusAction = SteamVR_Events.InputFocusAction( OnInputFocus );

			if ( hoverSphereTransform == null )
			{
				hoverSphereTransform = this.transform;
			}

			applicationLostFocusObject = new GameObject( "_application_lost_focus" );
			applicationLostFocusObject.transform.parent = transform;
			applicationLostFocusObject.SetActive( false );
		}


		//-------------------------------------------------
		IEnumerator Start()
		{
			// save off player instance
			playerInstance = Player.instance;
			if ( !playerInstance )
			{
				Debug.LogError( "No player instance found in Hand Start()" );
			}

			// allocate array for colliders
			overlappingColliders = new Collider[ColliderArraySize];

			// We are a "no SteamVR fallback hand" if we have this camera set
			// we'll use the right mouse to look around and left mouse to interact
			// - don't need to find the device
			if ( noSteamVRFallbackCamera )
			{
				yield break;
			}

			//Debug.Log( "Hand - initializing connection routine" );

			// Acquire the correct device index for the hand we want to be
			// Also for the other hand if we get there first
			while ( true )
			{
				// Don't need to run this every frame
				yield return new WaitForSeconds( 1.0f );

				// We have a controller now, break out of the loop!
				if ( controller != null )
					break;

				//Debug.Log( "Hand - checking controllers..." );

				// Initialize both hands simultaneously
				if ( startingHandType == HandType.Left || startingHandType == HandType.Right )
				{
					// Left/right relationship.
					// Wait until we have a clear unique left-right relationship to initialize.
					int leftIndex = SteamVR_Controller.GetDeviceIndex( SteamVR_Controller.DeviceRelation.Leftmost );
					int rightIndex = SteamVR_Controller.GetDeviceIndex( SteamVR_Controller.DeviceRelation.Rightmost );
					if ( leftIndex == -1 || rightIndex == -1 || leftIndex == rightIndex )
					{
						//Debug.Log( string.Format( "...Left/right hand relationship not yet established: leftIndex={0}, rightIndex={1}", leftIndex, rightIndex ) );
						continue;
					}

					int myIndex = ( startingHandType == HandType.Right ) ? rightIndex : leftIndex;
					int otherIndex = ( startingHandType == HandType.Right ) ? leftIndex : rightIndex;

					InitController( myIndex );
					if ( otherHand )
					{
						otherHand.InitController( otherIndex );
					}
				}
				else
				{
					// No left/right relationship. Just wait for a connection

					var vr = SteamVR.instance;
					for ( int i = 0; i < Valve.VR.OpenVR.k_unMaxTrackedDeviceCount; i++ )
					{
						if ( vr.hmd.GetTrackedDeviceClass( (uint)i ) != Valve.VR.ETrackedDeviceClass.Controller )
						{
							//Debug.Log( string.Format( "Hand - device {0} is not a controller", i ) );
							continue;
						}

						var device = SteamVR_Controller.Input( i );
						if ( !device.valid )
						{
							//Debug.Log( string.Format( "Hand - device {0} is not valid", i ) );
							continue;
						}

						if ( ( otherHand != null ) && ( otherHand.controller != null ) )
						{
							// Other hand is using this index, so we cannot use it.
							if ( i == (int)otherHand.controller.index )
							{
								//Debug.Log( string.Format( "Hand - device {0} is owned by the other hand", i ) );
								continue;
							}
						}

						InitController( i );
					}
				}
			}
		}


		//-------------------------------------------------
		private void UpdateHovering()
		{
			if ( ( noSteamVRFallbackCamera == null ) && ( controller == null ) )
			{
				return;
			}

			if ( hoverLocked )
				return;

			if ( applicationLostFocusObject.activeSelf )
				return;

			float closestDistance = float.MaxValue;
			Interactable closestInteractable = null;

			// Pick the closest hovering
			float flHoverRadiusScale = playerInstance.transform.lossyScale.x;
			float flScaledSphereRadius = hoverSphereRadius * flHoverRadiusScale;

			// if we're close to the floor, increase the radius to make things easier to pick up
			float handDiff = Mathf.Abs( transform.position.y - playerInstance.trackingOriginTransform.position.y );
			float boxMult = Util.RemapNumberClamped( handDiff, 0.0f, 0.5f * flHoverRadiusScale, 5.0f, 1.0f ) * flHoverRadiusScale;

			// null out old vals
			for ( int i = 0; i < overlappingColliders.Length; ++i )
			{
				overlappingColliders[i] = null;
			}

			Physics.OverlapBoxNonAlloc(
				hoverSphereTransform.position - new Vector3( 0, flScaledSphereRadius * boxMult - flScaledSphereRadius, 0 ),
				new Vector3( flScaledSphereRadius, flScaledSphereRadius * boxMult * 2.0f, flScaledSphereRadius ),
				overlappingColliders,
				Quaternion.identity,
				hoverLayerMask.value
			);

			// DebugVar
			int iActualColliderCount = 0;

			foreach ( Collider collider in overlappingColliders )
			{
				if ( collider == null )
					continue;

				Interactable contacting = collider.GetComponentInParent<Interactable>();

				// Yeah, it's null, skip
				if ( contacting == null )
					continue;

				// Ignore this collider for hovering
				IgnoreHovering ignore = collider.GetComponent<IgnoreHovering>();
				if ( ignore != null )
				{
					if ( ignore.onlyIgnoreHand == null || ignore.onlyIgnoreHand == this )
					{
						continue;
					}
				}

				// Can't hover over the object if it's attached
				if ( attachedObjects.FindIndex( l => l.attachedObject == contacting.gameObject ) != -1 )
					continue;

				// Occupied by another hand, so we can't touch it
				if ( otherHand && otherHand.hoveringInteractable == contacting )
					continue;

				// Best candidate so far...
				float distance = Vector3.Distance( contacting.transform.position, hoverSphereTransform.position );
				if ( distance < closestDistance )
				{
					closestDistance = distance;
					closestInteractable = contacting;
				}
				iActualColliderCount++;
			}

			// Hover on this one
			hoveringInteractable = closestInteractable;

			if ( iActualColliderCount > 0 && iActualColliderCount != prevOverlappingColliders )
			{
				prevOverlappingColliders = iActualColliderCount;
				HandDebugLog( "Found " + iActualColliderCount + " overlapping colliders." );
			}
		}


		//-------------------------------------------------
		private void UpdateNoSteamVRFallback()
		{
			if ( noSteamVRFallbackCamera )
			{
				Ray ray = noSteamVRFallbackCamera.ScreenPointToRay( Input.mousePosition );

				if ( attachedObjects.Count > 0 )
				{
					// Holding down the mouse:
					// move around a fixed distance from the camera
					transform.position = ray.origin + noSteamVRFallbackInteractorDistance * ray.direction;
				}
				else
				{
					// Not holding down the mouse:
					// cast out a ray to see what we should mouse over

					// Don't want to hit the hand and anything underneath it
					// So move it back behind the camera when we do the raycast
					Vector3 oldPosition = transform.position;
					transform.position = noSteamVRFallbackCamera.transform.forward * ( -1000.0f );

					RaycastHit raycastHit;
					if ( Physics.Raycast( ray, out raycastHit, noSteamVRFallbackMaxDistanceNoItem ) )
					{
						transform.position = raycastHit.point;

						// Remember this distance in case we click and drag the mouse
						noSteamVRFallbackInteractorDistance = Mathf.Min( noSteamVRFallbackMaxDistanceNoItem, raycastHit.distance );
					}
					else if ( noSteamVRFallbackInteractorDistance > 0.0f )
					{
						// Move it around at the distance we last had a hit
						transform.position = ray.origin + Mathf.Min( noSteamVRFallbackMaxDistanceNoItem, noSteamVRFallbackInteractorDistance ) * ray.direction;
					}
					else
					{
						// Didn't hit, just leave it where it was
						transform.position = oldPosition;
					}
				}
			}
		}


		//-------------------------------------------------
		private void UpdateDebugText()
		{
			if ( showDebugText )
			{
				if ( debugText == null )
				{
					debugText = new GameObject( "_debug_text" ).AddComponent<TextMesh>();
					debugText.fontSize = 120;
					debugText.characterSize = 0.001f;
					debugText.transform.parent = transform;

					debugText.transform.localRotation = Quaternion.Euler( 90.0f, 0.0f, 0.0f );
				}

				if ( GuessCurrentHandType() == HandType.Right )
				{
					debugText.transform.localPosition = new Vector3( -0.05f, 0.0f, 0.0f );
					debugText.alignment = TextAlignment.Right;
					debugText.anchor = TextAnchor.UpperRight;
				}
				else
				{
					debugText.transform.localPosition = new Vector3( 0.05f, 0.0f, 0.0f );
					debugText.alignment = TextAlignment.Left;
					debugText.anchor = TextAnchor.UpperLeft;
				}

				debugText.text = string.Format(
					"Hovering: {0}\n" +
					"Hover Lock: {1}\n" +
					"Attached: {2}\n" +
					"Total Attached: {3}\n" +
					"Type: {4}\n",
					( hoveringInteractable ? hoveringInteractable.gameObject.name : "null" ),
					hoverLocked,
					( currentAttachedObject ? currentAttachedObject.name : "null" ),
					attachedObjects.Count,
					GuessCurrentHandType().ToString() );
			}
			else
			{
				if ( debugText != null )
				{
					Destroy( debugText.gameObject );
				}
			}
		}


		//-------------------------------------------------
		void OnEnable()
		{
			inputFocusAction.enabled = true;

			// Stagger updates between hands
			float hoverUpdateBegin = ( ( otherHand != null ) && ( otherHand.GetInstanceID() < GetInstanceID() ) ) ? ( 0.5f * hoverUpdateInterval ) : ( 0.0f );
			InvokeRepeating( "UpdateHovering", hoverUpdateBegin, hoverUpdateInterval );
			InvokeRepeating( "UpdateDebugText", hoverUpdateBegin, hoverUpdateInterval );
		}


		//-------------------------------------------------
		void OnDisable()
		{
			inputFocusAction.enabled = false;

			CancelInvoke();
		}


		//-------------------------------------------------
		void Update()
		{
			UpdateNoSteamVRFallback();

			GameObject attached = currentAttachedObject;
			if ( attached )
			{
				attached.SendMessage( "HandAttachedUpdate", this, SendMessageOptions.DontRequireReceiver );
			}

			if ( hoveringInteractable )
			{
				hoveringInteractable.SendMessage( "HandHoverUpdate", this, SendMessageOptions.DontRequireReceiver );
			}
		}


		//-------------------------------------------------
		void LateUpdate()
		{
			//Re-attach the controller if nothing else is attached to the hand
			if ( controllerObject != null && attachedObjects.Count == 0 )
			{
				AttachObject( controllerObject );
			}
		}


		//-------------------------------------------------
		private void OnInputFocus( bool hasFocus )
		{
			if ( hasFocus )
			{
				DetachObject( applicationLostFocusObject, true );
				applicationLostFocusObject.SetActive( false );
				UpdateHandPoses();
				UpdateHovering();
				BroadcastMessage( "OnParentHandInputFocusAcquired", SendMessageOptions.DontRequireReceiver );
			}
			else
			{
				applicationLostFocusObject.SetActive( true );
				AttachObject( applicationLostFocusObject, AttachmentFlags.ParentToHand );
				BroadcastMessage( "OnParentHandInputFocusLost", SendMessageOptions.DontRequireReceiver );
			}
		}


		//-------------------------------------------------
		void FixedUpdate()
		{
			UpdateHandPoses();
		}


		//-------------------------------------------------
		void OnDrawGizmos()
		{
			Gizmos.color = new Color( 0.5f, 1.0f, 0.5f, 0.9f );
			Transform sphereTransform = hoverSphereTransform ? hoverSphereTransform : this.transform;
			Gizmos.DrawWireSphere( sphereTransform.position, hoverSphereRadius );
		}


		//-------------------------------------------------
		private void HandDebugLog( string msg )
		{
			if ( spewDebugText )
			{
				Debug.Log( "Hand (" + this.name + "): " + msg );
			}
		}


		//-------------------------------------------------
		private void UpdateHandPoses()
		{
			if ( controller != null )
			{
				SteamVR vr = SteamVR.instance;
				if ( vr != null )
				{
					var pose = new Valve.VR.TrackedDevicePose_t();
					var gamePose = new Valve.VR.TrackedDevicePose_t();
					var err = vr.compositor.GetLastPoseForTrackedDeviceIndex( controller.index, ref pose, ref gamePose );
					if ( err == Valve.VR.EVRCompositorError.None )
					{
						var t = new SteamVR_Utils.RigidTransform( gamePose.mDeviceToAbsoluteTracking );
						transform.localPosition = t.pos;
						transform.localRotation = t.rot;
					}
				}
			}
		}


		//-------------------------------------------------
		// Continue to hover over this object indefinitely, whether or not the Hand moves out of its interaction trigger volume.
		//
		// interactable - The Interactable to hover over indefinitely.
		//-------------------------------------------------
		public void HoverLock( Interactable interactable )
		{
			HandDebugLog( "HoverLock " + interactable );
			hoverLocked = true;
			hoveringInteractable = interactable;
		}


		//-------------------------------------------------
		// Stop hovering over this object indefinitely.
		//
		// interactable - The hover-locked Interactable to stop hovering over indefinitely.
		//-------------------------------------------------
		public void HoverUnlock( Interactable interactable )
		{
			HandDebugLog( "HoverUnlock " + interactable );
			if ( hoveringInteractable == interactable )
			{
				hoverLocked = false;
			}
		}

		//-------------------------------------------------
		// Was the standard interaction button just pressed? In VR, this is a trigger press. In 2D fallback, this is a mouse left-click.
		//-------------------------------------------------
		public bool GetStandardInteractionButtonDown()
		{
			if ( noSteamVRFallbackCamera )
			{
				return Input.GetMouseButtonDown( 0 );
			}
			else if ( controller != null )
			{
				return controller.GetHairTriggerDown();
			}

			return false;
		}


		//-------------------------------------------------
		// Was the standard interaction button just released? In VR, this is a trigger press. In 2D fallback, this is a mouse left-click.
		//-------------------------------------------------
		public bool GetStandardInteractionButtonUp()
		{
			if ( noSteamVRFallbackCamera )
			{
				return Input.GetMouseButtonUp( 0 );
			}
			else if ( controller != null )
			{
				return controller.GetHairTriggerUp();
			}

			return false;
		}


		//-------------------------------------------------
		// Is the standard interaction button being pressed? In VR, this is a trigger press. In 2D fallback, this is a mouse left-click.
		//-------------------------------------------------
		public bool GetStandardInteractionButton()
		{
			if ( noSteamVRFallbackCamera )
			{
				return Input.GetMouseButton( 0 );
			}
			else if ( controller != null )
			{
				return controller.GetHairTrigger();
			}

			return false;
		}


		//-------------------------------------------------
		private void InitController( int index )
		{
			if ( controller == null )
			{
				controller = SteamVR_Controller.Input( index );

				HandDebugLog( "Hand " + name + " connected with device index " + controller.index );

				controllerObject = GameObject.Instantiate( controllerPrefab );
				controllerObject.SetActive( true );
				controllerObject.name = controllerPrefab.name + "_" + this.name;
				controllerObject.layer = gameObject.layer;
				controllerObject.tag = gameObject.tag;
				AttachObject( controllerObject );
				controller.TriggerHapticPulse( 800 );

				// If the player's scale has been changed the object to attach will be the wrong size.
				// To fix this we change the object's scale back to its original, pre-attach scale.
				controllerObject.transform.localScale = controllerPrefab.transform.localScale;

				this.BroadcastMessage( "OnHandInitialized", index, SendMessageOptions.DontRequireReceiver ); // let child objects know we've initialized
			}
		}
	}

#if UNITY_EDITOR
	//-------------------------------------------------------------------------
	[UnityEditor.CustomEditor( typeof( Hand ) )]
	public class HandEditor : UnityEditor.Editor
	{
		//-------------------------------------------------
		// Custom Inspector GUI allows us to click from within the UI
		//-------------------------------------------------
		public override void OnInspectorGUI()
		{
			DrawDefaultInspector();

			Hand hand = (Hand)target;

			if ( hand.otherHand )
			{
				if ( hand.otherHand.otherHand != hand )
				{
					UnityEditor.EditorGUILayout.HelpBox( "The otherHand of this Hand's otherHand is not this Hand.", UnityEditor.MessageType.Warning );
				}

				if ( hand.startingHandType == Hand.HandType.Left && hand.otherHand.startingHandType != Hand.HandType.Right )
				{
					UnityEditor.EditorGUILayout.HelpBox( "This is a left Hand but otherHand is not a right Hand.", UnityEditor.MessageType.Warning );
				}

				if ( hand.startingHandType == Hand.HandType.Right && hand.otherHand.startingHandType != Hand.HandType.Left )
				{
					UnityEditor.EditorGUILayout.HelpBox( "This is a right Hand but otherHand is not a left Hand.", UnityEditor.MessageType.Warning );
				}

				if ( hand.startingHandType == Hand.HandType.Any && hand.otherHand.startingHandType != Hand.HandType.Any )
				{
					UnityEditor.EditorGUILayout.HelpBox( "This is an any-handed Hand but otherHand is not an any-handed Hand.", UnityEditor.MessageType.Warning );
				}
			}
		}
	}
#endif
}
