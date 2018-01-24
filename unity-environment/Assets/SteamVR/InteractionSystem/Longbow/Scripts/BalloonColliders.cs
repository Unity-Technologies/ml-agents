//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Can be attached to the controller to collide with the balloons
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class BalloonColliders : MonoBehaviour
	{
		public GameObject[] colliders;
		private Vector3[] colliderLocalPositions;
		private Quaternion[] colliderLocalRotations;

		private Rigidbody rb;

		//-------------------------------------------------
		void Awake()
		{
			rb = GetComponent<Rigidbody>();

			colliderLocalPositions = new Vector3[colliders.Length];
			colliderLocalRotations = new Quaternion[colliders.Length];

			for ( int i = 0; i < colliders.Length; ++i )
			{
				colliderLocalPositions[i] = colliders[i].transform.localPosition;
				colliderLocalRotations[i] = colliders[i].transform.localRotation;

				colliders[i].name = gameObject.name + "." + colliders[i].name;
			}
		}


		//-------------------------------------------------
		void OnEnable()
		{
			for ( int i = 0; i < colliders.Length; ++i )
			{
				colliders[i].transform.SetParent( transform );

				colliders[i].transform.localPosition = colliderLocalPositions[i];
				colliders[i].transform.localRotation = colliderLocalRotations[i];

				colliders[i].transform.SetParent( null );

				FixedJoint fixedJoint = colliders[i].AddComponent<FixedJoint>();
				fixedJoint.connectedBody = rb;
				fixedJoint.breakForce = Mathf.Infinity;
				fixedJoint.breakTorque = Mathf.Infinity;
				fixedJoint.enableCollision = false;
				fixedJoint.enablePreprocessing = true;

				colliders[i].SetActive( true );
			}
		}


		//-------------------------------------------------
		void OnDisable()
		{
			for ( int i = 0; i < colliders.Length; ++i )
			{
				if ( colliders[i] != null )
				{
					Destroy( colliders[i].GetComponent<FixedJoint>() );

					colliders[i].SetActive( false );
				}
			}
		}


		//-------------------------------------------------
		void OnDestroy()
		{
			for ( int i = 0; i < colliders.Length; ++i )
			{
				Destroy( colliders[i] );
			}
		}
	}
}
