//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Adds SteamVR render support to existing camera objects
//
//=============================================================================

using UnityEngine;
using System.Collections;
using System.Reflection;
using Valve.VR;

[RequireComponent(typeof(Camera))]
public class SteamVR_Camera : MonoBehaviour
{
	[SerializeField]
	private Transform _head;
	public Transform head { get { return _head; } }
	public Transform offset { get { return _head; } } // legacy
	public Transform origin { get { return _head.parent; } }

	public new Camera camera { get; private set; }

	[SerializeField]
	private Transform _ears;
	public Transform ears { get { return _ears; } }

	public Ray GetRay()
	{
		return new Ray(_head.position, _head.forward);
	}

	public bool wireframe = false;

	static public float sceneResolutionScale
	{
		get { return UnityEngine.VR.VRSettings.renderScale; }
		set { UnityEngine.VR.VRSettings.renderScale = value; }
	}

	#region Enable / Disable

	void OnDisable()
	{
		SteamVR_Render.Remove(this);
	}

	void OnEnable()
	{
		// Bail if no hmd is connected
		var vr = SteamVR.instance;
		if (vr == null)
		{
			if (head != null)
			{
				head.GetComponent<SteamVR_TrackedObject>().enabled = false;
			}

			enabled = false;
			return;
		}

		// Convert camera rig for native OpenVR integration.
		var t = transform;
		if (head != t)
		{
			Expand();

			t.parent = origin;

			while (head.childCount > 0)
				head.GetChild(0).parent = t;

			// Keep the head around, but parent to the camera now since it moves with the hmd
			// but existing content may still have references to this object.
			head.parent = t;
			head.localPosition = Vector3.zero;
			head.localRotation = Quaternion.identity;
			head.localScale = Vector3.one;
			head.gameObject.SetActive(false);

			_head = t;
		}

		if (ears == null)
		{
			var e = transform.GetComponentInChildren<SteamVR_Ears>();
			if (e != null)
				_ears = e.transform;
        }

		if (ears != null)
			ears.GetComponent<SteamVR_Ears>().vrcam = this;

		SteamVR_Render.Add(this);
	}

	#endregion

	#region Functionality to ensure SteamVR_Camera component is always the last component on an object

	void Awake()
	{
		camera = GetComponent<Camera>(); // cached to avoid runtime lookup
		ForceLast();
    }

	static Hashtable values;

	public void ForceLast()
	{
		if (values != null)
		{
			// Restore values on new instance
			foreach (DictionaryEntry entry in values)
			{
				var f = entry.Key as FieldInfo;
				f.SetValue(this, entry.Value);
			}
			values = null;
		}
		else
		{
			// Make sure it's the last component
			var components = GetComponents<Component>();

			// But first make sure there aren't any other SteamVR_Cameras on this object.
			for (int i = 0; i < components.Length; i++)
			{
				var c = components[i] as SteamVR_Camera;
				if (c != null && c != this)
				{
					DestroyImmediate(c);
				}
			}

			components = GetComponents<Component>();

			if (this != components[components.Length - 1])
			{
				// Store off values to be restored on new instance
				values = new Hashtable();
				var fields = GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
				foreach (var f in fields)
					if (f.IsPublic || f.IsDefined(typeof(SerializeField), true))
						values[f] = f.GetValue(this);

				var go = gameObject;
				DestroyImmediate(this);
				go.AddComponent<SteamVR_Camera>().ForceLast();
			}
		}
	}

	#endregion

	#region Expand / Collapse object hierarchy

#if UNITY_EDITOR
	public bool isExpanded { get { return head != null && transform.parent == head; } }
#endif
	const string eyeSuffix = " (eye)";
	const string earsSuffix = " (ears)";
	const string headSuffix = " (head)";
	const string originSuffix = " (origin)";
	public string baseName { get { return name.EndsWith(eyeSuffix) ? name.Substring(0, name.Length - eyeSuffix.Length) : name; } }

	// Object hierarchy creation to make it easy to parent other objects appropriately,
	// otherwise this gets called on demand at runtime. Remaining initialization is
	// performed at startup, once the hmd has been identified.
	public void Expand()
	{
		var _origin = transform.parent;
		if (_origin == null)
		{
			_origin = new GameObject(name + originSuffix).transform;
			_origin.localPosition = transform.localPosition;
			_origin.localRotation = transform.localRotation;
			_origin.localScale = transform.localScale;
		}

		if (head == null)
		{
			_head = new GameObject(name + headSuffix, typeof(SteamVR_TrackedObject)).transform;
			head.parent = _origin;
			head.position = transform.position;
			head.rotation = transform.rotation;
			head.localScale = Vector3.one;
			head.tag = tag;
		}

		if (transform.parent != head)
		{
			transform.parent = head;
			transform.localPosition = Vector3.zero;
			transform.localRotation = Quaternion.identity;
			transform.localScale = Vector3.one;

			while (transform.childCount > 0)
				transform.GetChild(0).parent = head;

			var guiLayer = GetComponent<GUILayer>();
			if (guiLayer != null)
			{
				DestroyImmediate(guiLayer);
				head.gameObject.AddComponent<GUILayer>();
			}

			var audioListener = GetComponent<AudioListener>();
			if (audioListener != null)
			{
				DestroyImmediate(audioListener);
				_ears = new GameObject(name + earsSuffix, typeof(SteamVR_Ears)).transform;
				ears.parent = _head;
				ears.localPosition = Vector3.zero;
				ears.localRotation = Quaternion.identity;
				ears.localScale = Vector3.one;
			}
		}

		if (!name.EndsWith(eyeSuffix))
			name += eyeSuffix;
	}

	public void Collapse()
	{
		transform.parent = null;

		// Move children and components from head back to camera.
		while (head.childCount > 0)
			head.GetChild(0).parent = transform;

		var guiLayer = head.GetComponent<GUILayer>();
		if (guiLayer != null)
		{
			DestroyImmediate(guiLayer);
			gameObject.AddComponent<GUILayer>();
		}

		if (ears != null)
		{
			while (ears.childCount > 0)
				ears.GetChild(0).parent = transform;

			DestroyImmediate(ears.gameObject);
			_ears = null;

			gameObject.AddComponent(typeof(AudioListener));
		}

		if (origin != null)
		{
			// If we created the origin originally, destroy it now.
			if (origin.name.EndsWith(originSuffix))
			{
				// Reparent any children so we don't accidentally delete them.
				var _origin = origin;
				while (_origin.childCount > 0)
					_origin.GetChild(0).parent = _origin.parent;

				DestroyImmediate(_origin.gameObject);
			}
			else
			{
				transform.parent = origin;
			}
		}

		DestroyImmediate(head.gameObject);
		_head = null;

		if (name.EndsWith(eyeSuffix))
			name = name.Substring(0, name.Length - eyeSuffix.Length);
	}

	#endregion
}

