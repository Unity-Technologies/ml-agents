//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: A package of items that can interact with the hands and be returned
//
//=============================================================================

using UnityEngine;
using System.Collections;

namespace Valve.VR.InteractionSystem
{
	//-------------------------------------------------------------------------
	public class ItemPackage : MonoBehaviour
	{
		public enum ItemPackageType { Unrestricted, OneHanded, TwoHanded }

		public new string name;
		public ItemPackageType packageType = ItemPackageType.Unrestricted;
		public GameObject itemPrefab; // object to be spawned on tracked controller
		public GameObject otherHandItemPrefab; // object to be spawned in Other Hand
		public GameObject previewPrefab; // used to preview inputObject
		public GameObject fadedPreviewPrefab; // used to preview insubstantial inputObject
	}
}
