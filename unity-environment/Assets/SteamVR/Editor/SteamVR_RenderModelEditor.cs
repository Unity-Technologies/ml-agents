//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: Custom inspector display for SteamVR_RenderModel
//
//=============================================================================

using UnityEngine;
using UnityEditor;
using System.Text;
using System.Collections.Generic;

[CustomEditor(typeof(SteamVR_RenderModel)), CanEditMultipleObjects]
public class SteamVR_RenderModelEditor : Editor
{
	SerializedProperty script, index, modelOverride, shader, verbose, createComponents, updateDynamically;

	static string[] renderModelNames;
	int renderModelIndex;

	void OnEnable()
	{
		script = serializedObject.FindProperty("m_Script");
		index = serializedObject.FindProperty("index");
		modelOverride = serializedObject.FindProperty("modelOverride");
		shader = serializedObject.FindProperty("shader");
		verbose = serializedObject.FindProperty("verbose");
		createComponents = serializedObject.FindProperty("createComponents");
		updateDynamically = serializedObject.FindProperty("updateDynamically");

		// Load render model names if necessary.
		if (renderModelNames == null)
		{
			renderModelNames = LoadRenderModelNames();
		}

		// Update renderModelIndex based on current modelOverride value.
		if (modelOverride.stringValue != "")
		{
			for (int i = 0; i < renderModelNames.Length; i++)
			{
				if (modelOverride.stringValue == renderModelNames[i])
				{
					renderModelIndex = i;
					break;
				}
			}
		}
	}

	static string[] LoadRenderModelNames()
	{
		var results = new List<string>();
		results.Add("None");

		using (var holder = new SteamVR_RenderModel.RenderModelInterfaceHolder())
		{
			var renderModels = holder.instance;
			if (renderModels != null)
			{
				uint count = renderModels.GetRenderModelCount();
				for (uint i = 0; i < count; i++)
				{
					var buffer = new StringBuilder();
					var requiredSize = renderModels.GetRenderModelName(i, buffer, 0);
					if (requiredSize == 0)
						continue;

					buffer.EnsureCapacity((int)requiredSize);
					renderModels.GetRenderModelName(i, buffer, requiredSize);
					results.Add(buffer.ToString());
				}
			}
		}

		return results.ToArray();
	}

	public override void OnInspectorGUI()
	{
		serializedObject.Update();

		EditorGUILayout.PropertyField(script);
		EditorGUILayout.PropertyField(index);
		//EditorGUILayout.PropertyField(modelOverride);

		GUILayout.BeginHorizontal();
		GUILayout.Label(new GUIContent("Model Override", SteamVR_RenderModel.modelOverrideWarning));
		var selected = EditorGUILayout.Popup(renderModelIndex, renderModelNames);
		if (selected != renderModelIndex)
		{
			renderModelIndex = selected;
			modelOverride.stringValue = (selected > 0) ? renderModelNames[selected] : "";
		}
		GUILayout.EndHorizontal();

		EditorGUILayout.PropertyField(shader);
		EditorGUILayout.PropertyField(verbose);
		EditorGUILayout.PropertyField(createComponents);
		EditorGUILayout.PropertyField(updateDynamically);

		serializedObject.ApplyModifiedProperties();
	}
}

