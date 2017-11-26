//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Interactable that can be used to move in a circular motion
//
//=============================================================================

using UnityEngine;
using UnityEngine.Events;
using System.Collections;

namespace Valve.VR.InteractionSystem
{

	//-------------------------------------------------------------------------
	[RequireComponent( typeof( Interactable ) )]
	public class CircularDrive : MonoBehaviour
	{
		public enum Axis_t
		{
			XAxis,
			YAxis,
			ZAxis
		};

		[Tooltip( "The axis around which the circular drive will rotate in local space" )]
		public Axis_t axisOfRotation = Axis_t.XAxis;

		[Tooltip( "Child GameObject which has the Collider component to initiate interaction, only needs to be set if there is more than one Collider child" )]
		public Collider childCollider = null;

		[Tooltip( "A LinearMapping component to drive, if not specified one will be dynamically added to this GameObject" )]
		public LinearMapping linearMapping;

		[Tooltip( "If true, the drive will stay manipulating as long as the button is held down, if false, it will stop if the controller moves out of the collider" )]
		public bool hoverLock = false;

		[HeaderAttribute( "Limited Rotation" )]
		[Tooltip( "If true, the rotation will be limited to [minAngle, maxAngle], if false, the rotation is unlimited" )]
		public bool limited = false;
		public Vector2 frozenDistanceMinMaxThreshold = new Vector2( 0.1f, 0.2f );
		public UnityEvent onFrozenDistanceThreshold;

		[HeaderAttribute( "Limited Rotation Min" )]
		[Tooltip( "If limited is true, the specifies the lower limit, otherwise value is unused" )]
		public float minAngle = -45.0f;
		[Tooltip( "If limited, set whether drive will freeze its angle when the min angle is reached" )]
		public bool freezeOnMin = false;
		[Tooltip( "If limited, event invoked when minAngle is reached" )]
		public UnityEvent onMinAngle;

		[HeaderAttribute( "Limited Rotation Max" )]
		[Tooltip( "If limited is true, the specifies the upper limit, otherwise value is unused" )]
		public float maxAngle = 45.0f;
		[Tooltip( "If limited, set whether drive will freeze its angle when the max angle is reached" )]
		public bool freezeOnMax = false;
		[Tooltip( "If limited, event invoked when maxAngle is reached" )]
		public UnityEvent onMaxAngle;

		[Tooltip( "If limited is true, this forces the starting angle to be startAngle, clamped to [minAngle, maxAngle]" )]
		public bool forceStart = false;
		[Tooltip( "If limited is true and forceStart is true, the starting angle will be this, clamped to [minAngle, maxAngle]" )]
		public float startAngle = 0.0f;

		[Tooltip( "If true, the transform of the GameObject this component is on will be rotated accordingly" )]
		public bool rotateGameObject = true;

		[Tooltip( "If true, the path of the Hand (red) and the projected value (green) will be drawn" )]
		public bool debugPath = false;
		[Tooltip( "If debugPath is true, this is the maximum number of GameObjects to create to draw the path" )]
		public int dbgPathLimit = 50;

		[Tooltip( "If not null, the TextMesh will display the linear value and the angular value of this circular drive" )]
		public TextMesh debugText = null;

		[Tooltip( "The output angle value of the drive in degrees, unlimited will increase or decrease without bound, take the 360 modulus to find number of rotations" )]
		public float outAngle;

		private Quaternion start;

		private Vector3 worldPlaneNormal = new Vector3( 1.0f, 0.0f, 0.0f );
		private Vector3 localPlaneNormal = new Vector3( 1.0f, 0.0f, 0.0f );

		private Vector3 lastHandProjected;

		private Color red = new Color( 1.0f, 0.0f, 0.0f );
		private Color green = new Color( 0.0f, 1.0f, 0.0f );

		private GameObject[] dbgHandObjects;
		private GameObject[] dbgProjObjects;
		private GameObject dbgObjectsParent;
		private int dbgObjectCount = 0;
		private int dbgObjectIndex = 0;

		private bool driving = false;

		// If the drive is limited as is at min/max, angles greater than this are ignored 
		private float minMaxAngularThreshold = 1.0f;

		private bool frozen = false;
		private float frozenAngle = 0.0f;
		private Vector3 frozenHandWorldPos = new Vector3( 0.0f, 0.0f, 0.0f );
		private Vector2 frozenSqDistanceMinMaxThreshold = new Vector2( 0.0f, 0.0f );

		Hand handHoverLocked = null;

		//-------------------------------------------------
		private void Freeze( Hand hand )
		{
			frozen = true;
			frozenAngle = outAngle;
			frozenHandWorldPos = hand.hoverSphereTransform.position;
			frozenSqDistanceMinMaxThreshold.x = frozenDistanceMinMaxThreshold.x * frozenDistanceMinMaxThreshold.x;
			frozenSqDistanceMinMaxThreshold.y = frozenDistanceMinMaxThreshold.y * frozenDistanceMinMaxThreshold.y;
		}


		//-------------------------------------------------
		private void UnFreeze()
		{
			frozen = false;
			frozenHandWorldPos.Set( 0.0f, 0.0f, 0.0f );
		}


		//-------------------------------------------------
		void Start()
		{
			if ( childCollider == null )
			{
				childCollider = GetComponentInChildren<Collider>();
			}

			if ( linearMapping == null )
			{
				linearMapping = GetComponent<LinearMapping>();
			}

			if ( linearMapping == null )
			{
				linearMapping = gameObject.AddComponent<LinearMapping>();
			}

			worldPlaneNormal = new Vector3( 0.0f, 0.0f, 0.0f );
			worldPlaneNormal[(int)axisOfRotation] = 1.0f;

			localPlaneNormal = worldPlaneNormal;

			if ( transform.parent )
			{
				worldPlaneNormal = transform.parent.localToWorldMatrix.MultiplyVector( worldPlaneNormal ).normalized;
			}

			if ( limited )
			{
				start = Quaternion.identity;
				outAngle = transform.localEulerAngles[(int)axisOfRotation];

				if ( forceStart )
				{
					outAngle = Mathf.Clamp( startAngle, minAngle, maxAngle );
				}
			}
			else
			{
				start = Quaternion.AngleAxis( transform.localEulerAngles[(int)axisOfRotation], localPlaneNormal );
				outAngle = 0.0f;
			}

			if ( debugText )
			{
				debugText.alignment = TextAlignment.Left;
				debugText.anchor = TextAnchor.UpperLeft;
			}

			UpdateAll();
		}


		//-------------------------------------------------
		void OnDisable()
		{
			if ( handHoverLocked )
			{
				ControllerButtonHints.HideButtonHint( handHoverLocked, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
				handHoverLocked.HoverUnlock( GetComponent<Interactable>() );
				handHoverLocked = null;
			}
		}


		//-------------------------------------------------
		private IEnumerator HapticPulses( SteamVR_Controller.Device controller, float flMagnitude, int nCount )
		{
			if ( controller != null )
			{
				int nRangeMax = (int)Util.RemapNumberClamped( flMagnitude, 0.0f, 1.0f, 100.0f, 900.0f );
				nCount = Mathf.Clamp( nCount, 1, 10 );

				for ( ushort i = 0; i < nCount; ++i )
				{
					ushort duration = (ushort)Random.Range( 100, nRangeMax );
					controller.TriggerHapticPulse( duration );
					yield return new WaitForSeconds( .01f );
				}
			}
		}


		//-------------------------------------------------
		private void OnHandHoverBegin( Hand hand )
		{
			ControllerButtonHints.ShowButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
		}


		//-------------------------------------------------
		private void OnHandHoverEnd( Hand hand )
		{
			ControllerButtonHints.HideButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );

			if ( driving && hand.GetStandardInteractionButton() )
			{
				StartCoroutine( HapticPulses( hand.controller, 1.0f, 10 ) );
			}

			driving = false;
			handHoverLocked = null;
		}


		//-------------------------------------------------
		private void HandHoverUpdate( Hand hand )
		{
			if ( hand.GetStandardInteractionButtonDown() )
			{
				// Trigger was just pressed
				lastHandProjected = ComputeToTransformProjected( hand.hoverSphereTransform );

				if ( hoverLock )
				{
					hand.HoverLock( GetComponent<Interactable>() );
					handHoverLocked = hand;
				}

				driving = true;

				ComputeAngle( hand );
				UpdateAll();

				ControllerButtonHints.HideButtonHint( hand, Valve.VR.EVRButtonId.k_EButton_SteamVR_Trigger );
			}
			else if ( hand.GetStandardInteractionButtonUp() )
			{
				// Trigger was just released
				if ( hoverLock )
				{
					hand.HoverUnlock( GetComponent<Interactable>() );
					handHoverLocked = null;
				}
			}
			else if ( driving && hand.GetStandardInteractionButton() && hand.hoveringInteractable == GetComponent<Interactable>() )
			{
				ComputeAngle( hand );
				UpdateAll();
			}
		}


		//-------------------------------------------------
		private Vector3 ComputeToTransformProjected( Transform xForm )
		{
			Vector3 toTransform = ( xForm.position - transform.position ).normalized;
			Vector3 toTransformProjected = new Vector3( 0.0f, 0.0f, 0.0f );

			// Need a non-zero distance from the hand to the center of the CircularDrive
			if ( toTransform.sqrMagnitude > 0.0f )
			{
				toTransformProjected = Vector3.ProjectOnPlane( toTransform, worldPlaneNormal ).normalized;
			}
			else
			{
				Debug.LogFormat( "The collider needs to be a minimum distance away from the CircularDrive GameObject {0}", gameObject.ToString() );
				Debug.Assert( false, string.Format( "The collider needs to be a minimum distance away from the CircularDrive GameObject {0}", gameObject.ToString() ) );
			}

			if ( debugPath && dbgPathLimit > 0 )
			{
				DrawDebugPath( xForm, toTransformProjected );
			}

			return toTransformProjected;
		}


		//-------------------------------------------------
		private void DrawDebugPath( Transform xForm, Vector3 toTransformProjected )
		{
			if ( dbgObjectCount == 0 )
			{
				dbgObjectsParent = new GameObject( "Circular Drive Debug" );
				dbgHandObjects = new GameObject[dbgPathLimit];
				dbgProjObjects = new GameObject[dbgPathLimit];
				dbgObjectCount = dbgPathLimit;
				dbgObjectIndex = 0;
			}

			//Actual path
			GameObject gSphere = null;

			if ( dbgHandObjects[dbgObjectIndex] )
			{
				gSphere = dbgHandObjects[dbgObjectIndex];
			}
			else
			{
				gSphere = GameObject.CreatePrimitive( PrimitiveType.Sphere );
				gSphere.transform.SetParent( dbgObjectsParent.transform );
				dbgHandObjects[dbgObjectIndex] = gSphere;
			}

			gSphere.name = string.Format( "actual_{0}", (int)( ( 1.0f - red.r ) * 10.0f ) );
			gSphere.transform.position = xForm.position;
			gSphere.transform.rotation = Quaternion.Euler( 0.0f, 0.0f, 0.0f );
			gSphere.transform.localScale = new Vector3( 0.004f, 0.004f, 0.004f );
			gSphere.gameObject.GetComponent<Renderer>().material.color = red;

			if ( red.r > 0.1f )
			{
				red.r -= 0.1f;
			}
			else
			{
				red.r = 1.0f;
			}

			//Projected path
			gSphere = null;

			if ( dbgProjObjects[dbgObjectIndex] )
			{
				gSphere = dbgProjObjects[dbgObjectIndex];
			}
			else
			{
				gSphere = GameObject.CreatePrimitive( PrimitiveType.Sphere );
				gSphere.transform.SetParent( dbgObjectsParent.transform );
				dbgProjObjects[dbgObjectIndex] = gSphere;
			}

			gSphere.name = string.Format( "projed_{0}", (int)( ( 1.0f - green.g ) * 10.0f ) );
			gSphere.transform.position = transform.position + toTransformProjected * 0.25f;
			gSphere.transform.rotation = Quaternion.Euler( 0.0f, 0.0f, 0.0f );
			gSphere.transform.localScale = new Vector3( 0.004f, 0.004f, 0.004f );
			gSphere.gameObject.GetComponent<Renderer>().material.color = green;

			if ( green.g > 0.1f )
			{
				green.g -= 0.1f;
			}
			else
			{
				green.g = 1.0f;
			}

			dbgObjectIndex = ( dbgObjectIndex + 1 ) % dbgObjectCount;
		}


		//-------------------------------------------------
		// Updates the LinearMapping value from the angle
		//-------------------------------------------------
		private void UpdateLinearMapping()
		{
			if ( limited )
			{
				// Map it to a [0, 1] value
				linearMapping.value = ( outAngle - minAngle ) / ( maxAngle - minAngle );
			}
			else
			{
				// Normalize to [0, 1] based on 360 degree windings
				float flTmp = outAngle / 360.0f;
				linearMapping.value = flTmp - Mathf.Floor( flTmp );
			}

			UpdateDebugText();
		}


		//-------------------------------------------------
		// Updates the LinearMapping value from the angle
		//-------------------------------------------------
		private void UpdateGameObject()
		{
			if ( rotateGameObject )
			{
				transform.localRotation = start * Quaternion.AngleAxis( outAngle, localPlaneNormal );
			}
		}


		//-------------------------------------------------
		// Updates the Debug TextMesh with the linear mapping value and the angle
		//-------------------------------------------------
		private void UpdateDebugText()
		{
			if ( debugText )
			{
				debugText.text = string.Format( "Linear: {0}\nAngle:  {1}\n", linearMapping.value, outAngle );
			}
		}


		//-------------------------------------------------
		// Updates the Debug TextMesh with the linear mapping value and the angle
		//-------------------------------------------------
		private void UpdateAll()
		{
			UpdateLinearMapping();
			UpdateGameObject();
			UpdateDebugText();
		}


		//-------------------------------------------------
		// Computes the angle to rotate the game object based on the change in the transform
		//-------------------------------------------------
		private void ComputeAngle( Hand hand )
		{
			Vector3 toHandProjected = ComputeToTransformProjected( hand.hoverSphereTransform );

			if ( !toHandProjected.Equals( lastHandProjected ) )
			{
				float absAngleDelta = Vector3.Angle( lastHandProjected, toHandProjected );

				if ( absAngleDelta > 0.0f )
				{
					if ( frozen )
					{
						float frozenSqDist = ( hand.hoverSphereTransform.position - frozenHandWorldPos ).sqrMagnitude;
						if ( frozenSqDist > frozenSqDistanceMinMaxThreshold.x )
						{
							outAngle = frozenAngle + Random.Range( -1.0f, 1.0f );

							float magnitude = Util.RemapNumberClamped( frozenSqDist, frozenSqDistanceMinMaxThreshold.x, frozenSqDistanceMinMaxThreshold.y, 0.0f, 1.0f );
							if ( magnitude > 0 )
							{
								StartCoroutine( HapticPulses( hand.controller, magnitude, 10 ) );
							}
							else
							{
								StartCoroutine( HapticPulses( hand.controller, 0.5f, 10 ) );
							}

							if ( frozenSqDist >= frozenSqDistanceMinMaxThreshold.y )
							{
								onFrozenDistanceThreshold.Invoke();
							}
						}
					}
					else
					{
						Vector3 cross = Vector3.Cross( lastHandProjected, toHandProjected ).normalized;
						float dot = Vector3.Dot( worldPlaneNormal, cross );

						float signedAngleDelta = absAngleDelta;

						if ( dot < 0.0f )
						{
							signedAngleDelta = -signedAngleDelta;
						}

						if ( limited )
						{
							float angleTmp = Mathf.Clamp( outAngle + signedAngleDelta, minAngle, maxAngle );

							if ( outAngle == minAngle )
							{
								if ( angleTmp > minAngle && absAngleDelta < minMaxAngularThreshold )
								{
									outAngle = angleTmp;
									lastHandProjected = toHandProjected;
								}
							}
							else if ( outAngle == maxAngle )
							{
								if ( angleTmp < maxAngle && absAngleDelta < minMaxAngularThreshold )
								{
									outAngle = angleTmp;
									lastHandProjected = toHandProjected;
								}
							}
							else if ( angleTmp == minAngle )
							{
								outAngle = angleTmp;
								lastHandProjected = toHandProjected;
								onMinAngle.Invoke();
								if ( freezeOnMin )
								{
									Freeze( hand );
								}
							}
							else if ( angleTmp == maxAngle )
							{
								outAngle = angleTmp;
								lastHandProjected = toHandProjected;
								onMaxAngle.Invoke();
								if ( freezeOnMax )
								{
									Freeze( hand );
								}
							}
							else
							{
								outAngle = angleTmp;
								lastHandProjected = toHandProjected;
							}
						}
						else
						{
							outAngle += signedAngleDelta;
							lastHandProjected = toHandProjected;
						}
					}
				}
			}
		}
	}
}
