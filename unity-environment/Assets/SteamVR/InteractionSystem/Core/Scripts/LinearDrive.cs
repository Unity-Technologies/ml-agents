//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Drives a linear mapping based on position between 2 positions
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	[RequireComponent( typeof( Interactable ) )]
	public class LinearDrive : MonoBehaviour
	{
		public Transform startPosition;
		public Transform endPosition;
		public LinearMapping linearMapping;
		public bool repositionGameObject = true;
		public bool maintainMomemntum = true;
		public float momemtumDampenRate = 5.0f;

		private float initialMappingOffset;
		private int numMappingChangeSamples = 5;
		private float[] mappingChangeSamples;
		private float prevMapping = 0.0f;
		private float mappingChangeRate;
		private int sampleCount = 0;


		//-------------------------------------------------
		void Awake()
		{
			mappingChangeSamples = new float[numMappingChangeSamples];
		}


		//-------------------------------------------------
		void Start()
		{
			if ( linearMapping == null )
			{
				linearMapping = GetComponent<LinearMapping>();
			}

			if ( linearMapping == null )
			{
				linearMapping = gameObject.AddComponent<LinearMapping>();
			}

            initialMappingOffset = linearMapping.value;

			if ( repositionGameObject )
			{
				UpdateLinearMapping( transform );
			}
		}


		//-------------------------------------------------
		private void HandHoverUpdate( Hand hand )
		{
			if ( hand.GetStandardInteractionButtonDown() )
			{
				hand.HoverLock( GetComponent<Interactable>() );

				initialMappingOffset = linearMapping.value - CalculateLinearMapping( hand.transform );
				sampleCount = 0;
				mappingChangeRate = 0.0f;
			}

			if ( hand.GetStandardInteractionButtonUp() )
			{
				hand.HoverUnlock( GetComponent<Interactable>() );

				CalculateMappingChangeRate();
			}

			if ( hand.GetStandardInteractionButton() )
			{
				UpdateLinearMapping( hand.transform );
			}
		}


		//-------------------------------------------------
		private void CalculateMappingChangeRate()
		{
			//Compute the mapping change rate
			mappingChangeRate = 0.0f;
			int mappingSamplesCount = Mathf.Min( sampleCount, mappingChangeSamples.Length );
			if ( mappingSamplesCount != 0 )
			{
				for ( int i = 0; i < mappingSamplesCount; ++i )
				{
					mappingChangeRate += mappingChangeSamples[i];
				}
				mappingChangeRate /= mappingSamplesCount;
			}
		}


		//-------------------------------------------------
		private void UpdateLinearMapping( Transform tr )
		{
			prevMapping = linearMapping.value;
			linearMapping.value = Mathf.Clamp01( initialMappingOffset + CalculateLinearMapping( tr ) );

			mappingChangeSamples[sampleCount % mappingChangeSamples.Length] = ( 1.0f / Time.deltaTime ) * ( linearMapping.value - prevMapping );
			sampleCount++;

			if ( repositionGameObject )
			{
				transform.position = Vector3.Lerp( startPosition.position, endPosition.position, linearMapping.value );
			}
		}


		//-------------------------------------------------
		private float CalculateLinearMapping( Transform tr )
		{
			Vector3 direction = endPosition.position - startPosition.position;
			float length = direction.magnitude;
			direction.Normalize();

			Vector3 displacement = tr.position - startPosition.position;

			return Vector3.Dot( displacement, direction ) / length;
		}


		//-------------------------------------------------
		void Update()
		{
			if ( maintainMomemntum && mappingChangeRate != 0.0f )
			{
				//Dampen the mapping change rate and apply it to the mapping
				mappingChangeRate = Mathf.Lerp( mappingChangeRate, 0.0f, momemtumDampenRate * Time.deltaTime );
				linearMapping.value = Mathf.Clamp01( linearMapping.value + ( mappingChangeRate * Time.deltaTime ) );

				if ( repositionGameObject )
				{
					transform.position = Vector3.Lerp( startPosition.position, endPosition.position, linearMapping.value );
				}
			}
		}
	}
}
