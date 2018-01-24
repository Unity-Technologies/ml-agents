//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Single location that the player can teleport to
//
//=============================================================================

using UnityEngine;
using UnityEngine.UI;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class TeleportPoint : TeleportMarkerBase
	{
		public enum TeleportPointType
		{
			MoveToLocation,
			SwitchToNewScene
		};

		//Public variables
		public TeleportPointType teleportType = TeleportPointType.MoveToLocation;
		public string title;
		public string switchToScene;
		public Color titleVisibleColor;
		public Color titleHighlightedColor;
		public Color titleLockedColor;
		public bool playerSpawnPoint = false;

		//Private data
		private bool gotReleventComponents = false;
		private MeshRenderer markerMesh;
		private MeshRenderer switchSceneIcon;
		private MeshRenderer moveLocationIcon;
		private MeshRenderer lockedIcon;
		private MeshRenderer pointIcon;
		private Transform lookAtJointTransform;
		private new Animation animation;
		private Text titleText;
		private Player player;
		private Vector3 lookAtPosition = Vector3.zero;
		private int tintColorID = 0;
		private Color tintColor = Color.clear;
		private Color titleColor = Color.clear;
		private float fullTitleAlpha = 0.0f;

		//Constants
		private const string switchSceneAnimation = "switch_scenes_idle";
		private const string moveLocationAnimation = "move_location_idle";
		private const string lockedAnimation = "locked_idle";


		//-------------------------------------------------
		public override bool showReticle
		{
			get
			{
				return false;
			}
		}


		//-------------------------------------------------
		void Awake()
		{
			GetRelevantComponents();

			animation = GetComponent<Animation>();

			tintColorID = Shader.PropertyToID( "_TintColor" );

			moveLocationIcon.gameObject.SetActive( false );
			switchSceneIcon.gameObject.SetActive( false );
			lockedIcon.gameObject.SetActive( false );

			UpdateVisuals();
		}


		//-------------------------------------------------
		void Start()
		{
			player = Player.instance;
		}


		//-------------------------------------------------
		void Update()
		{
			if ( Application.isPlaying )
			{
				lookAtPosition.x = player.hmdTransform.position.x;
				lookAtPosition.y = lookAtJointTransform.position.y;
				lookAtPosition.z = player.hmdTransform.position.z;

				lookAtJointTransform.LookAt( lookAtPosition );
			}
		}


		//-------------------------------------------------
		public override bool ShouldActivate( Vector3 playerPosition )
		{
			return ( Vector3.Distance( transform.position, playerPosition ) > 1.0f );
		}


		//-------------------------------------------------
		public override bool ShouldMovePlayer()
		{
			return true;
		}


		//-------------------------------------------------
		public override void Highlight( bool highlight )
		{
			if ( !locked )
			{
				if ( highlight )
				{
					SetMeshMaterials( Teleport.instance.pointHighlightedMaterial, titleHighlightedColor );
				}
				else
				{
					SetMeshMaterials( Teleport.instance.pointVisibleMaterial, titleVisibleColor );
				}
			}

			if ( highlight )
			{
				pointIcon.gameObject.SetActive( true );
				animation.Play();
			}
			else
			{
				pointIcon.gameObject.SetActive( false );
				animation.Stop();
			}
		}


		//-------------------------------------------------
		public override void UpdateVisuals()
		{
			if ( !gotReleventComponents )
			{
				return;
			}

			if ( locked )
			{
				SetMeshMaterials( Teleport.instance.pointLockedMaterial, titleLockedColor );

				pointIcon = lockedIcon;

				animation.clip = animation.GetClip( lockedAnimation );
			}
			else
			{
				SetMeshMaterials( Teleport.instance.pointVisibleMaterial, titleVisibleColor );

				switch ( teleportType )
				{
					case TeleportPointType.MoveToLocation:
						{
							pointIcon = moveLocationIcon;

							animation.clip = animation.GetClip( moveLocationAnimation );
						}
						break;
					case TeleportPointType.SwitchToNewScene:
						{
							pointIcon = switchSceneIcon;

							animation.clip = animation.GetClip( switchSceneAnimation );
						}
						break;
				}
			}

			titleText.text = title;
		}


		//-------------------------------------------------
		public override void SetAlpha( float tintAlpha, float alphaPercent )
		{
			tintColor = markerMesh.material.GetColor( tintColorID );
			tintColor.a = tintAlpha;

			markerMesh.material.SetColor( tintColorID, tintColor );
			switchSceneIcon.material.SetColor( tintColorID, tintColor );
			moveLocationIcon.material.SetColor( tintColorID, tintColor );
			lockedIcon.material.SetColor( tintColorID, tintColor );

			titleColor.a = fullTitleAlpha * alphaPercent;
			titleText.color = titleColor;
		}


		//-------------------------------------------------
		public void SetMeshMaterials( Material material, Color textColor )
		{
			markerMesh.material = material;
			switchSceneIcon.material = material;
			moveLocationIcon.material = material;
			lockedIcon.material = material;

			titleColor = textColor;
			fullTitleAlpha = textColor.a;
			titleText.color = titleColor;
		}


		//-------------------------------------------------
		public void TeleportToScene()
		{
			if ( !string.IsNullOrEmpty( switchToScene ) )
			{
				Debug.Log( "TeleportPoint: Hook up your level loading logic to switch to new scene: " + switchToScene );
			}
			else
			{
				Debug.LogError( "TeleportPoint: Invalid scene name to switch to: " + switchToScene );
			}
		}


		//-------------------------------------------------
		public void GetRelevantComponents()
		{
			markerMesh = transform.Find( "teleport_marker_mesh" ).GetComponent<MeshRenderer>();
			switchSceneIcon = transform.Find( "teleport_marker_lookat_joint/teleport_marker_icons/switch_scenes_icon" ).GetComponent<MeshRenderer>();
			moveLocationIcon = transform.Find( "teleport_marker_lookat_joint/teleport_marker_icons/move_location_icon" ).GetComponent<MeshRenderer>();
			lockedIcon = transform.Find( "teleport_marker_lookat_joint/teleport_marker_icons/locked_icon" ).GetComponent<MeshRenderer>();
			lookAtJointTransform = transform.Find( "teleport_marker_lookat_joint" );

			titleText = transform.Find( "teleport_marker_lookat_joint/teleport_marker_canvas/teleport_marker_canvas_text" ).GetComponent<Text>();

			gotReleventComponents = true;
		}


		//-------------------------------------------------
		public void ReleaseRelevantComponents()
		{
			markerMesh = null;
			switchSceneIcon = null;
			moveLocationIcon = null;
			lockedIcon = null;
			lookAtJointTransform = null;
			titleText = null;
		}


		//-------------------------------------------------
		public void UpdateVisualsInEditor()
		{
			if ( Application.isPlaying )
			{
				return;
			}

			GetRelevantComponents();

			if ( locked )
			{
				lockedIcon.gameObject.SetActive( true );
				moveLocationIcon.gameObject.SetActive( false );
				switchSceneIcon.gameObject.SetActive( false );

				markerMesh.sharedMaterial = Teleport.instance.pointLockedMaterial;
				lockedIcon.sharedMaterial = Teleport.instance.pointLockedMaterial;

				titleText.color = titleLockedColor;
			}
			else
			{
				lockedIcon.gameObject.SetActive( false );

				markerMesh.sharedMaterial = Teleport.instance.pointVisibleMaterial;
				switchSceneIcon.sharedMaterial = Teleport.instance.pointVisibleMaterial;
				moveLocationIcon.sharedMaterial = Teleport.instance.pointVisibleMaterial;

				titleText.color = titleVisibleColor;

				switch ( teleportType )
				{
					case TeleportPointType.MoveToLocation:
						{
							moveLocationIcon.gameObject.SetActive( true );
							switchSceneIcon.gameObject.SetActive( false );
						}
						break;
					case TeleportPointType.SwitchToNewScene:
						{
							moveLocationIcon.gameObject.SetActive( false );
							switchSceneIcon.gameObject.SetActive( true );
						}
						break;
				}
			}

			titleText.text = title;

			ReleaseRelevantComponents();
		}
	}


#if UNITY_EDITOR
	//-------------------------------------------------------------------------
	[CustomEditor( typeof( TeleportPoint ) )]
	public class TeleportPointEditor : Editor
	{
		//-------------------------------------------------
		void OnEnable()
		{
			if ( Selection.activeTransform )
			{
				TeleportPoint teleportPoint = Selection.activeTransform.GetComponent<TeleportPoint>();
				teleportPoint.UpdateVisualsInEditor();
			}
		}


		//-------------------------------------------------
		public override void OnInspectorGUI()
		{
			DrawDefaultInspector();

			if ( Selection.activeTransform )
			{
				TeleportPoint teleportPoint = Selection.activeTransform.GetComponent<TeleportPoint>();
				if ( GUI.changed )
				{
					teleportPoint.UpdateVisualsInEditor();
				}
			}
		}
	}
#endif
}
