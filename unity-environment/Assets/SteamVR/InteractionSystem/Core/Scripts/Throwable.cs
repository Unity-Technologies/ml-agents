//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Basic throwable object
//
//=============================================================================

using UnityEngine;
using UnityEngine.Events;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	[RequireComponent( typeof( Interactable ) )]
	[RequireComponent( typeof( Rigidbody ) )]
	[RequireComponent( typeof( VelocityEstimator ) )]
	public class Throwable : MonoBehaviour
	{
		[EnumFlags]
		[Tooltip( "The flags used to attach this object to the hand." )]
		public Hand.AttachmentFlags attachmentFlags = Hand.AttachmentFlags.ParentToHand | Hand.AttachmentFlags.DetachFromOtherHand;

		[Tooltip( "Name of the attachment transform under in the hand's hierarchy which the object should should snap to." )]
		public string attachmentPoint;

		[Tooltip( "How fast must this object be moving to attach due to a trigger hold instead of a trigger press?" )]
		public float catchSpeedThreshold = 0.0f;

		[Tooltip( "When detaching the object, should it return to its original parent?" )]
		public bool restoreOriginalParent = false;

		public bool attachEaseIn = false;
		public AnimationCurve snapAttachEaseInCurve = AnimationCurve.EaseInOut( 0.0f, 0.0f, 1.0f, 1.0f );
		public float snapAttachEaseInTime = 0.15f;
		public string[] attachEaseInAttachmentNames;

		private VelocityEstimator velocityEstimator;
		private bool attached = false;
		private float attachTime;
		private Vector3 attachPosition;
		private Quaternion attachRotation;
		private Transform attachEaseInTransform;

		public UnityEvent onPickUp;
		public UnityEvent onDetachFromHand;

		public bool snapAttachEaseInCompleted = false;


		//-------------------------------------------------
		void Awake()
		{
			velocityEstimator = GetComponent<VelocityEstimator>();

			if ( attachEaseIn )
			{
				attachmentFlags &= ~Hand.AttachmentFlags.SnapOnAttach;
			}

			Rigidbody rb = GetComponent<Rigidbody>();
			rb.maxAngularVelocity = 50.0f;
		}


		//-------------------------------------------------
		private void OnHandHoverBegin( Hand hand )
		{
			bool showHint = false;

			// "Catch" the throwable by holding down the interaction button instead of pressing it.
			// Only do this if the throwable is moving faster than the prescribed threshold speed,
			// and if it isn't attached to another hand
			if ( !attached )
			{
				if ( hand.GetStandardInteractionButton() )
				{
					Rigidbody rb = GetComponent<Rigidbody>();
					if ( rb.velocity.magnitude >= catchSpeedThreshold )
					{
						hand.AttachObject( gameObject, attachmentFlags, attachmentPoint );
						showHint = false;
					}
				}
			}

			if ( showHint )
			{
				ControllerButtonHints.ShowButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
			}
		}


		//-------------------------------------------------
		private void OnHandHoverEnd( Hand hand )
		{
			ControllerButtonHints.HideButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
		}


		//-------------------------------------------------
		private void HandHoverUpdate( Hand hand )
		{
			//Trigger got pressed
			if ( hand.GetStandardInteractionButtonDown() )
			{
				hand.AttachObject( gameObject, attachmentFlags, attachmentPoint );
				ControllerButtonHints.HideButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
			}
		}

		//-------------------------------------------------
		private void OnAttachedToHand( Hand hand )
		{
			attached = true;

			onPickUp.Invoke();

			hand.HoverLock( null );

			Rigidbody rb = GetComponent<Rigidbody>();
			rb.isKinematic = true;
			rb.interpolation = RigidbodyInterpolation.None;

			if ( hand.controller == null )
			{
				velocityEstimator.BeginEstimatingVelocity();
			}

			attachTime = Time.time;
			attachPosition = transform.position;
			attachRotation = transform.rotation;

			if ( attachEaseIn )
			{
				attachEaseInTransform = hand.transform;
				if ( !Util.IsNullOrEmpty( attachEaseInAttachmentNames ) )
				{
					float smallestAngle = float.MaxValue;
					for ( int i = 0; i < attachEaseInAttachmentNames.Length; i++ )
					{
						Transform t = hand.GetAttachmentTransform( attachEaseInAttachmentNames[i] );
						float angle = Quaternion.Angle( t.rotation, attachRotation );
						if ( angle < smallestAngle )
						{
							attachEaseInTransform = t;
							smallestAngle = angle;
						}
					}
				}
			}

			snapAttachEaseInCompleted = false;
		}


		//-------------------------------------------------
		private void OnDetachedFromHand( Hand hand )
		{
			attached = false;

			onDetachFromHand.Invoke();

			hand.HoverUnlock( null );

			Rigidbody rb = GetComponent<Rigidbody>();
			rb.isKinematic = false;
			rb.interpolation = RigidbodyInterpolation.Interpolate;

			Vector3 position = Vector3.zero;
			Vector3 velocity = Vector3.zero;
			Vector3 angularVelocity = Vector3.zero;
			if ( hand.controller == null )
			{
				velocityEstimator.FinishEstimatingVelocity();
				velocity = velocityEstimator.GetVelocityEstimate();
				angularVelocity = velocityEstimator.GetAngularVelocityEstimate();
				position = velocityEstimator.transform.position;
			}
			else
			{
				velocity = Player.instance.trackingOriginTransform.TransformVector( hand.controller.velocity );
				angularVelocity = Player.instance.trackingOriginTransform.TransformVector( hand.controller.angularVelocity );
				position = hand.transform.position;
			}

			Vector3 r = transform.TransformPoint( rb.centerOfMass ) - position;
			rb.velocity = velocity + Vector3.Cross( angularVelocity, r );
			rb.angularVelocity = angularVelocity;

			// Make the object travel at the release velocity for the amount
			// of time it will take until the next fixed update, at which
			// point Unity physics will take over
			float timeUntilFixedUpdate = ( Time.fixedDeltaTime + Time.fixedTime ) - Time.time;
			transform.position += timeUntilFixedUpdate * velocity;
			float angle = Mathf.Rad2Deg * angularVelocity.magnitude;
			Vector3 axis = angularVelocity.normalized;
			transform.rotation *= Quaternion.AngleAxis( angle * timeUntilFixedUpdate, axis );
		}


		//-------------------------------------------------
		private void HandAttachedUpdate( Hand hand )
		{
			//Trigger got released
			if ( !hand.GetStandardInteractionButton() )
			{
				// Detach ourselves late in the frame.
				// This is so that any vehicles the player is attached to
				// have a chance to finish updating themselves.
				// If we detach now, our position could be behind what it
				// will be at the end of the frame, and the object may appear
				// to teleport behind the hand when the player releases it.
				StartCoroutine( LateDetach( hand ) );
			}

			if ( attachEaseIn )
			{
				float t = Util.RemapNumberClamped( Time.time, attachTime, attachTime + snapAttachEaseInTime, 0.0f, 1.0f );
				if ( t < 1.0f )
				{
					t = snapAttachEaseInCurve.Evaluate( t );
					transform.position = Vector3.Lerp( attachPosition, attachEaseInTransform.position, t );
					transform.rotation = Quaternion.Lerp( attachRotation, attachEaseInTransform.rotation, t );
				}
				else if ( !snapAttachEaseInCompleted )
				{
					gameObject.SendMessage( "OnThrowableAttachEaseInCompleted", hand, SendMessageOptions.DontRequireReceiver );
					snapAttachEaseInCompleted = true;
				}
			}
		}


		//-------------------------------------------------
		private IEnumerator LateDetach( Hand hand )
		{
			yield return new WaitForEndOfFrame();

			hand.DetachObject( gameObject, restoreOriginalParent );
		}


		//-------------------------------------------------
		private void OnHandFocusAcquired( Hand hand )
		{
			gameObject.SetActive( true );
			velocityEstimator.BeginEstimatingVelocity();
		}


		//-------------------------------------------------
		private void OnHandFocusLost( Hand hand )
		{
			gameObject.SetActive( false );
			velocityEstimator.FinishEstimatingVelocity();
		}
	}
}
