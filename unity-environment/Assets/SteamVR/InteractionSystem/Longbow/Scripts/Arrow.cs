//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: The arrow for the longbow
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class Arrow : MonoBehaviour
	{
		public ParticleSystem glintParticle;
		public Rigidbody arrowHeadRB;
		public Rigidbody shaftRB;

		public PhysicMaterial targetPhysMaterial;

		private Vector3 prevPosition;
		private Quaternion prevRotation;
		private Vector3 prevVelocity;
		private Vector3 prevHeadPosition;

		public SoundPlayOneshot fireReleaseSound;
		public SoundPlayOneshot airReleaseSound;
		public SoundPlayOneshot hitTargetSound;

		public PlaySound hitGroundSound;

		private bool inFlight;
		private bool released;
		private bool hasSpreadFire = false;

		private int travelledFrames = 0;

		private GameObject scaleParentObject = null;


		//-------------------------------------------------
		void Start()
		{
			Physics.IgnoreCollision( shaftRB.GetComponent<Collider>(), Player.instance.headCollider );
		}


		//-------------------------------------------------
		void FixedUpdate()
		{
			if ( released && inFlight )
			{
				prevPosition = transform.position;
				prevRotation = transform.rotation;
				prevVelocity = GetComponent<Rigidbody>().velocity;
				prevHeadPosition = arrowHeadRB.transform.position;
				travelledFrames++;
			}
		}


		//-------------------------------------------------
		public void ArrowReleased( float inputVelocity )
		{
			inFlight = true;
			released = true;

			airReleaseSound.Play();

			if ( glintParticle != null )
			{
				glintParticle.Play();
			}

			if ( gameObject.GetComponentInChildren<FireSource>().isBurning )
			{
				fireReleaseSound.Play();
			}

			// Check if arrow is shot inside or too close to an object
			RaycastHit[] hits = Physics.SphereCastAll( transform.position, 0.01f, transform.forward, 0.80f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore );
			foreach ( RaycastHit hit in hits )
			{
				if ( hit.collider.gameObject != gameObject && hit.collider.gameObject != arrowHeadRB.gameObject && hit.collider != Player.instance.headCollider )
				{
					Destroy( gameObject );
					return;
				}
			}

			travelledFrames = 0;
			prevPosition = transform.position;
			prevRotation = transform.rotation;
			prevHeadPosition = arrowHeadRB.transform.position;
			prevVelocity = GetComponent<Rigidbody>().velocity;

			Destroy( gameObject, 30 );
		}


		//-------------------------------------------------
		void OnCollisionEnter( Collision collision )
		{
			if ( inFlight )
			{
				Rigidbody rb = GetComponent<Rigidbody>();
				float rbSpeed = rb.velocity.sqrMagnitude;
				bool canStick = ( targetPhysMaterial != null && collision.collider.sharedMaterial == targetPhysMaterial && rbSpeed > 0.2f );
				bool hitBalloon = collision.collider.gameObject.GetComponent<Balloon>() != null;

				if ( travelledFrames < 2 && !canStick )
				{
					// Reset transform but halve your velocity
					transform.position = prevPosition - prevVelocity * Time.deltaTime;
					transform.rotation = prevRotation;

					Vector3 reflfectDir = Vector3.Reflect( arrowHeadRB.velocity, collision.contacts[0].normal );
					arrowHeadRB.velocity = reflfectDir * 0.25f;
					shaftRB.velocity = reflfectDir * 0.25f;

					travelledFrames = 0;
					return;
				}

				if ( glintParticle != null )
				{
					glintParticle.Stop( true );
				}

				// Only play hit sounds if we're moving quickly
				if ( rbSpeed > 0.1f )
				{
					hitGroundSound.Play();
				}

				FireSource arrowFire = gameObject.GetComponentInChildren<FireSource>();
				FireSource fireSourceOnTarget = collision.collider.GetComponentInParent<FireSource>();

				if ( arrowFire != null && arrowFire.isBurning && ( fireSourceOnTarget != null ) )
				{
					if ( !hasSpreadFire )
					{
						collision.collider.gameObject.SendMessageUpwards( "FireExposure", gameObject, SendMessageOptions.DontRequireReceiver );
						hasSpreadFire = true;
					}
				}
				else
				{
					// Only count collisions with good speed so that arrows on the ground can't deal damage
					// always pop balloons
					if ( rbSpeed > 0.1f || hitBalloon )
					{
						collision.collider.gameObject.SendMessageUpwards( "ApplyDamage", SendMessageOptions.DontRequireReceiver );
						gameObject.SendMessage( "HasAppliedDamage", SendMessageOptions.DontRequireReceiver );
					}
				}

				if ( hitBalloon )
				{
					// Revert my physics properties cause I don't want balloons to influence my travel
					transform.position = prevPosition;
					transform.rotation = prevRotation;
					arrowHeadRB.velocity = prevVelocity;
					Physics.IgnoreCollision( arrowHeadRB.GetComponent<Collider>(), collision.collider );
					Physics.IgnoreCollision( shaftRB.GetComponent<Collider>(), collision.collider );
				}

				if ( canStick )
				{
					StickInTarget( collision, travelledFrames < 2 );
				}

				// Player Collision Check (self hit)
				if ( Player.instance && collision.collider == Player.instance.headCollider )
				{
					Player.instance.PlayerShotSelf();
				}
			}
		}


		//-------------------------------------------------
		private void StickInTarget( Collision collision, bool bSkipRayCast )
		{
			Vector3 prevForward = prevRotation * Vector3.forward;

			// Only stick in target if the collider is front of the arrow head
			if ( !bSkipRayCast )
			{
				RaycastHit[] hitInfo;
				hitInfo = Physics.RaycastAll( prevHeadPosition - prevVelocity * Time.deltaTime, prevForward, prevVelocity.magnitude * Time.deltaTime * 2.0f );
				bool properHit = false;
				for ( int i = 0; i < hitInfo.Length; ++i )
				{
					RaycastHit hit = hitInfo[i];

					if ( hit.collider == collision.collider )
					{
						properHit = true;
						break;
					}
				}

				if ( !properHit )
				{
					return;
				}
			}

			Destroy( glintParticle );

			inFlight = false;

			shaftRB.velocity = Vector3.zero;
			shaftRB.angularVelocity = Vector3.zero;
			shaftRB.isKinematic = true;
			shaftRB.useGravity = false;
			shaftRB.transform.GetComponent<BoxCollider>().enabled = false;

			arrowHeadRB.velocity = Vector3.zero;
			arrowHeadRB.angularVelocity = Vector3.zero;
			arrowHeadRB.isKinematic = true;
			arrowHeadRB.useGravity = false;
			arrowHeadRB.transform.GetComponent<BoxCollider>().enabled = false;

			hitTargetSound.Play();


			// If the hit item has a parent, dock an empty object to that
			// this fixes an issue with scaling hierarchy. I suspect this is not sustainable for a large object / scaling hierarchy.
			scaleParentObject = new GameObject( "Arrow Scale Parent" );
			Transform parentTransform = collision.collider.transform;

			// Don't do this for weebles because of how it has a fixed joint
			ExplosionWobble wobble = collision.collider.gameObject.GetComponent<ExplosionWobble>();
			if ( !wobble )
			{
				if ( parentTransform.parent )
				{
					parentTransform = parentTransform.parent;
				}
			}

			scaleParentObject.transform.parent = parentTransform;

			// Move the arrow to the place on the target collider we were expecting to hit prior to the impact itself knocking it around
			transform.parent = scaleParentObject.transform;
			transform.rotation = prevRotation;
			transform.position = prevPosition;
			transform.position = collision.contacts[0].point - transform.forward * ( 0.75f - ( Util.RemapNumberClamped( prevVelocity.magnitude, 0f, 10f, 0.0f, 0.1f ) + Random.Range( 0.0f, 0.05f ) ) );
		}


		//-------------------------------------------------
		void OnDestroy()
		{
			if ( scaleParentObject != null )
			{
				Destroy( scaleParentObject );
			}
		}
	}
}
