// // Unity C# reference source
// // Copyright (c) Unity Technologies. For terms of use, see
// // https://unity3d.com/legal/licenses/Unity_Reference_Only_License

// using UnityEngine;
// using UnityEditor;
// using System.Collections.Generic;
// using System.Linq;

// namespace UnityEditor
// {
//     [CustomEditor(typeof(AudioSource))]
//     [CanEditMultipleObjects]
//     class AudioSourceInspector : Editor
//     {
//         SerializedProperty m_AudioClip;
//         SerializedProperty m_PlayOnAwake;
//         SerializedProperty m_Volume;
//         SerializedProperty m_Pitch;
//         SerializedProperty m_Loop;
//         SerializedProperty m_Mute;
//         SerializedProperty m_Spatialize;
//         SerializedProperty m_SpatializePostEffects;
//         SerializedProperty m_Priority;
//         SerializedProperty m_DopplerLevel;
//         SerializedProperty m_MinDistance;
//         SerializedProperty m_MaxDistance;
//         SerializedProperty m_Pan2D;
//         SerializedProperty m_RolloffMode;
//         SerializedProperty m_BypassEffects;
//         SerializedProperty m_BypassListenerEffects;
//         SerializedProperty m_BypassReverbZones;
//         SerializedProperty m_OutputAudioMixerGroup;

//         SerializedObject m_LowpassObject;

//         class AudioCurveWrapper
//         {
//             public AudioCurveType type;
//             public GUIContent legend;
//             public int id;
//             public Color color;
//             public SerializedProperty curveProp;
//             public float rangeMin;
//             public float rangeMax;
//             public AudioCurveWrapper(AudioCurveType type, string legend, int id, Color color, SerializedProperty curveProp, float rangeMin, float rangeMax)
//             {
//                 this.type = type;
//                 this.legend = new GUIContent(legend);
//                 this.id = id;
//                 this.color = color;
//                 this.curveProp = curveProp;
//                 this.rangeMin = rangeMin;
//                 this.rangeMax = rangeMax;
//             }
//         }
//         private AudioCurveWrapper[] m_AudioCurves;

//         CurveEditor m_CurveEditor = null;
//         Vector3 m_LastSourcePosition;
//         Vector3 m_LastListenerPosition;

//         const int kRolloffCurveID = 0;
//         const int kSpatialBlendCurveID = 1;
//         const int kSpreadCurveID = 2;
//         const int kLowPassCurveID = 3;
//         const int kReverbZoneMixCurveID = 4;
//         internal const float kMaxCutoffFrequency = 22000.0f;
//         const float EPSILON = 0.0001f;


//         static CurveEditorSettings m_CurveEditorSettings = new CurveEditorSettings();
//         internal static Color kRolloffCurveColor  = new Color(0.90f, 0.30f, 0.20f, 1.0f);
//         internal static Color kSpatialCurveColor = new Color(0.25f, 0.70f, 0.20f, 1.0f);
//         internal static Color kSpreadCurveColor   = new Color(0.25f, 0.55f, 0.95f, 1.0f);
//         internal static Color kLowPassCurveColor  = new Color(0.80f, 0.25f, 0.90f, 1.0f);
//         internal static Color kReverbZoneMixCurveColor = new Color(0.70f, 0.70f, 0.20f, 1.0f);

//         internal bool[] m_SelectedCurves = new bool[0];

//         private enum AudioCurveType { Volume, SpatialBlend, Lowpass, Spread, ReverbZoneMix }

//         private bool m_Expanded3D = false;

//         internal static class Styles
//         {
//             public static GUIStyle labelStyle = "ProfilerBadge";
//             public static GUIContent rolloffLabel =  EditorGUIUtility.TrTextContent("Volume Rolloff", "Which type of rolloff curve to use");
//             public static string controlledByCurveLabel = "Controlled by curve";
//             public static GUIContent audioClipLabel = EditorGUIUtility.TrTextContent("AudioClip", "The AudioClip asset played by the AudioSource. Can be undefined if the AudioSource is generating a live stream of audio via OnAudioFilterRead.");
//             public static GUIContent panStereoLabel = EditorGUIUtility.TrTextContent("Stereo Pan", "Only valid for Mono and Stereo AudioClips. Mono sounds will be panned at constant power left and right. Stereo sounds will have each left/right value faded up and down according to the specified pan value.");
//             public static GUIContent spatialBlendLabel = EditorGUIUtility.TrTextContent("Spatial Blend", "Sets how much this AudioSource is treated as a 3D source. 3D sources are affected by spatial position and spread. If 3D Pan Level is 0, all spatial attenuation is ignored.");
//             public static GUIContent reverbZoneMixLabel = EditorGUIUtility.TrTextContent("Reverb Zone Mix", "Sets how much of the signal this AudioSource is mixing into the global reverb associated with the zones. [0, 1] is a linear range (like volume) while [1, 1.1] lets you boost the reverb mix by 10 dB.");
//             public static GUIContent dopplerLevelLabel = EditorGUIUtility.TrTextContent("Doppler Level", "Specifies how much the pitch is changed based on the relative velocity between AudioListener and AudioSource.");
//             public static GUIContent spreadLabel = EditorGUIUtility.TrTextContent("Spread", "Sets the spread of a 3d sound in speaker space");
//             public static GUIContent outputMixerGroupLabel = EditorGUIUtility.TrTextContent("Output", "Set whether the sound should play through an Audio Mixer first or directly to the Audio Listener");
//             public static GUIContent volumeLabel = EditorGUIUtility.TrTextContent("Volume", "Sets the overall volume of the sound.");
//             public static GUIContent pitchLabel = EditorGUIUtility.TrTextContent("Pitch", "Sets the frequency of the sound. Use this to slow down or speed up the sound.");
//             public static GUIContent priorityLabel = EditorGUIUtility.TrTextContent("Priority", "Sets the priority of the source. Note that a sound with a larger priority value will more likely be stolen by sounds with smaller priority values.");
//             public static GUIContent spatializeLabel = EditorGUIUtility.TrTextContent("Spatialize", "Enables or disables custom spatialization for the AudioSource.");
//             public static GUIContent spatializePostEffectsLabel = EditorGUIUtility.TrTextContent("Spatialize Post Effects", "Determines if the custom spatializer is applied before or after the effect filters attached to the AudioSource. This flag only has an effect if the spatialize flag is enabled on the AudioSource.");
//             public static GUIContent priorityLeftLabel = EditorGUIUtility.TrTextContent("High");
//             public static GUIContent priorityRightLabel = EditorGUIUtility.TrTextContent("Low");
//             public static GUIContent spatialLeftLabel = EditorGUIUtility.TrTextContent("2D");
//             public static GUIContent spatialRightLabel = EditorGUIUtility.TrTextContent("3D");
//             public static GUIContent panLeftLabel = EditorGUIUtility.TrTextContent("Left");
//             public static GUIContent panRightLabel = EditorGUIUtility.TrTextContent("Right");
//             public static string xAxisLabel = L10n.Tr("Distance");
//         }

//         Vector3 GetSourcePos(Object target)
//         {
//             AudioSource source = (AudioSource)target;
//             if (source == null)
//                 return new Vector3(0.0f, 0.0f, 0.0f);
//             return source.transform.position;
//         }

//         void OnEnable()
//         {
//             m_AudioClip = serializedObject.FindProperty("m_audioClip");
//             m_PlayOnAwake = serializedObject.FindProperty("m_PlayOnAwake");
//             m_Volume = serializedObject.FindProperty("m_Volume");
//             m_Pitch = serializedObject.FindProperty("m_Pitch");
//             m_Loop = serializedObject.FindProperty("Loop");
//             m_Mute = serializedObject.FindProperty("Mute");
//             m_Spatialize = serializedObject.FindProperty("Spatialize");
//             m_SpatializePostEffects = serializedObject.FindProperty("SpatializePostEffects");
//             m_Priority = serializedObject.FindProperty("Priority");
//             m_DopplerLevel = serializedObject.FindProperty("DopplerLevel");
//             m_MinDistance = serializedObject.FindProperty("MinDistance");
//             m_MaxDistance = serializedObject.FindProperty("MaxDistance");
//             m_Pan2D = serializedObject.FindProperty("Pan2D");
//             m_RolloffMode = serializedObject.FindProperty("rolloffMode");
//             m_BypassEffects = serializedObject.FindProperty("BypassEffects");
//             m_BypassListenerEffects = serializedObject.FindProperty("BypassListenerEffects");
//             m_BypassReverbZones = serializedObject.FindProperty("BypassReverbZones");
//             m_OutputAudioMixerGroup = serializedObject.FindProperty("OutputAudioMixerGroup");

//             m_AudioCurves = new AudioCurveWrapper[]
//             {
//                 new AudioCurveWrapper(AudioCurveType.Volume, "Volume", kRolloffCurveID, kRolloffCurveColor, serializedObject.FindProperty("rolloffCustomCurve"), 0, 1),
//                 new AudioCurveWrapper(AudioCurveType.SpatialBlend, "Spatial Blend", kSpatialBlendCurveID, kSpatialCurveColor, serializedObject.FindProperty("panLevelCustomCurve"), 0, 1),
//                 new AudioCurveWrapper(AudioCurveType.Spread, "Spread", kSpreadCurveID, kSpreadCurveColor, serializedObject.FindProperty("spreadCustomCurve"), 0, 1),
//                 new AudioCurveWrapper(AudioCurveType.Lowpass, "Low-Pass", kLowPassCurveID, kLowPassCurveColor, null, 0, 1),
//                 new AudioCurveWrapper(AudioCurveType.ReverbZoneMix, "Reverb Zone Mix", kReverbZoneMixCurveID, kReverbZoneMixCurveColor, serializedObject.FindProperty("reverbZoneMixCustomCurve"), 0, 1.1f)
//             };

//             m_CurveEditorSettings.hRangeMin = 0.0f;
//             m_CurveEditorSettings.vRangeMin = 0.0f;
//             m_CurveEditorSettings.vRangeMax = 1.1f;
//             m_CurveEditorSettings.hRangeMax = 1.0f;
//             m_CurveEditorSettings.vSlider = false;
//             m_CurveEditorSettings.hSlider = false;

//             TickStyle hTS = new TickStyle();
//             hTS.tickColor.color = new Color(0.0f, 0.0f, 0.0f, 0.15f);
//             hTS.distLabel = 30;
//             m_CurveEditorSettings.hTickStyle = hTS;
//             TickStyle vTS = new TickStyle();
//             vTS.tickColor.color = new Color(0.0f, 0.0f, 0.0f, 0.15f);
//             vTS.distLabel = 20;
//             m_CurveEditorSettings.vTickStyle = vTS;

//             m_CurveEditorSettings.undoRedoSelection = true;

//             m_CurveEditor = new CurveEditor(new Rect(0, 0, 1000, 100), new CurveWrapper[0], false);
//             m_CurveEditor.settings = m_CurveEditorSettings;
//             m_CurveEditor.margin = 25;
//             m_CurveEditor.SetShownHRangeInsideMargins(0.0f, 1.0f);
//             m_CurveEditor.SetShownVRangeInsideMargins(0.0f, 1.1f);
//             m_CurveEditor.ignoreScrollWheelUntilClicked = true;

//             m_LastSourcePosition = GetSourcePos(target);
//             m_LastListenerPosition = AudioUtil.GetListenerPos();
//             EditorApplication.update += Update;

//             m_Expanded3D = EditorPrefs.GetBool("AudioSourceExpanded3D", m_Expanded3D);
//         }

//         void OnDisable()
//         {
//             m_CurveEditor.OnDisable();

//             EditorApplication.update -= Update;

//             EditorPrefs.SetBool("AudioSourceExpanded3D", m_Expanded3D);
//         }

//         CurveWrapper[] GetCurveWrapperArray()
//         {
//             List<CurveWrapper> wrappers = new List<CurveWrapper>();

//             foreach (AudioCurveWrapper audioCurve in m_AudioCurves)
//             {
//                 if (audioCurve.curveProp == null)
//                     continue;

//                 bool includeCurve = false;
//                 AnimationCurve curve = audioCurve.curveProp.animationCurveValue;

//                 // Special handling of volume rolloff curve
//                 if (audioCurve.type == AudioCurveType.Volume)
//                 {
//                     AudioRolloffMode mode = (AudioRolloffMode)m_RolloffMode.enumValueIndex;
//                     if (m_RolloffMode.hasMultipleDifferentValues)
//                     {
//                         includeCurve = false;
//                     }
//                     else if (mode == AudioRolloffMode.Custom)
//                     {
//                         includeCurve = !audioCurve.curveProp.hasMultipleDifferentValues;
//                     }
//                     else
//                     {
//                         includeCurve = !m_MinDistance.hasMultipleDifferentValues && !m_MaxDistance.hasMultipleDifferentValues;
//                         if (mode == AudioRolloffMode.Linear)
//                             curve = AnimationCurve.Linear(m_MinDistance.floatValue / m_MaxDistance.floatValue, 1.0f, 1.0f, 0.0f);
//                         else if (mode == AudioRolloffMode.Logarithmic)
//                             curve = Logarithmic(m_MinDistance.floatValue / m_MaxDistance.floatValue, 1.0f, 1.0f);
//                     }
//                 }
//                 // All other curves
//                 else
//                 {
//                     includeCurve = !audioCurve.curveProp.hasMultipleDifferentValues;
//                 }

//                 if (includeCurve)
//                 {
//                     if (curve.length == 0)
//                         Debug.LogError(audioCurve.legend.text + " curve has no keys!");
//                     else
//                         wrappers.Add(GetCurveWrapper(curve, audioCurve));
//                 }
//             }

//             return wrappers.ToArray();
//         }

//         private CurveWrapper GetCurveWrapper(AnimationCurve curve, AudioCurveWrapper audioCurve)
//         {
//             float colorMultiplier = !EditorGUIUtility.isProSkin ? 0.9f : 1.0f;
//             Color colorMult = new Color(colorMultiplier, colorMultiplier, colorMultiplier, 1);

//             CurveWrapper wrapper = new CurveWrapper();
//             wrapper.id = audioCurve.id;
//             wrapper.groupId = -1;
//             wrapper.color = audioCurve.color * colorMult;
//             wrapper.hidden = false;
//             wrapper.readOnly = false;
//             wrapper.renderer = new NormalCurveRenderer(curve);
//             wrapper.renderer.SetCustomRange(0.0f, 1.0f);
//             wrapper.getAxisUiScalarsCallback = GetAxisScalars;
//             wrapper.useScalingInKeyEditor = true;
//             wrapper.xAxisLabel = Styles.xAxisLabel;
//             wrapper.yAxisLabel = audioCurve.legend.text;
//             return wrapper;
//         }

//         // Callback for Curve Editor to get axis labels
//         public Vector2 GetAxisScalars()
//         {
//             return new Vector2(m_MaxDistance.floatValue, 1);
//         }

//         private static float LogarithmicValue(float distance, float minDistance, float rolloffScale)
//         {
//             if ((distance > minDistance) && (rolloffScale != 1.0f))
//             {
//                 distance -= minDistance;
//                 distance *= rolloffScale;
//                 distance += minDistance;
//             }
//             if (distance < .000001f)
//                 distance = .000001f;
//             return minDistance / distance;
//         }

//         /// A logarithmic curve starting at /timeStart/, /valueStart/ and ending at /timeEnd/, /valueEnd/
//         private static AnimationCurve Logarithmic(float timeStart, float timeEnd, float logBase)
//         {
//             float value, slope, s;
//             List<Keyframe> keys = new List<Keyframe>();
//             // Just plain set the step to 2 always. It can't really be any less,
//             // or the curvature will end up being imprecise in certain edge cases.
//             float step = 2;
//             timeStart = Mathf.Max(timeStart, 0.0001f);
//             for (float d = timeStart; d < timeEnd; d *= step)
//             {
//                 // Add key w. sensible tangents
//                 value = LogarithmicValue(d, timeStart, logBase);
//                 s = d / 50.0f;
//                 slope = (LogarithmicValue(d + s, timeStart, logBase) - LogarithmicValue(d - s, timeStart, logBase)) / (s * 2);
//                 keys.Add(new Keyframe(d, value, slope, slope));
//             }

//             // last key
//             value = LogarithmicValue(timeEnd, timeStart, logBase);
//             s = timeEnd / 50.0f;
//             slope = (LogarithmicValue(timeEnd + s, timeStart, logBase) - LogarithmicValue(timeEnd - s, timeStart, logBase)) / (s * 2);
//             keys.Add(new Keyframe(timeEnd, value, slope, slope));

//             return new AnimationCurve(keys.ToArray());
//         }

//         private void Update()
//         {
//             // listener moved?
//             Vector3 sourcePos = GetSourcePos(target);
//             Vector3 listenerPos = AudioUtil.GetListenerPos();
//             if ((m_LastSourcePosition - sourcePos).sqrMagnitude > EPSILON || (m_LastListenerPosition - listenerPos).sqrMagnitude > EPSILON)
//             {
//                 m_LastSourcePosition = sourcePos;
//                 m_LastListenerPosition = listenerPos;
//                 Repaint();
//             }
//         }

//         private void HandleLowPassFilter()
//         {
//             AudioCurveWrapper audioCurve = m_AudioCurves[kLowPassCurveID];

//             // Low pass filter present for all targets?
//             AudioLowPassFilter[] filterArray = new AudioLowPassFilter[targets.Length];
//             for (int i = 0; i < targets.Length; i++)
//             {
//                 filterArray[i] = ((AudioSource)targets[i]).GetComponent<AudioLowPassFilter>();
//                 if (filterArray[i] == null)
//                 {
//                     m_LowpassObject = null;
//                     audioCurve.curveProp = null;
//                     // Return if any of the GameObjects don't have an AudioLowPassFilter
//                     return;
//                 }
//             }

//             // All the GameObjects have an AudioLowPassFilter.
//             // If we don't have the corresponding SerializedObject and SerializedProperties, create them.
//             if (audioCurve.curveProp == null)
//             {
//                 m_LowpassObject = new SerializedObject(filterArray);
//                 audioCurve.curveProp = m_LowpassObject.FindProperty("lowpassLevelCustomCurve");
//             }
//         }

//         public override void OnInspectorGUI()
//         {
//             //Bug fix: 1018456 Moved the HandleLowPassFilter method before updating the serializedObjects
//             HandleLowPassFilter();

//             serializedObject.Update();

//             if (m_LowpassObject != null)
//                 m_LowpassObject.Update();


//             UpdateWrappersAndLegend();

//             EditorGUILayout.PropertyField(m_AudioClip, Styles.audioClipLabel);
//             EditorGUILayout.Space();
//             EditorGUILayout.PropertyField(m_OutputAudioMixerGroup, Styles.outputMixerGroupLabel);
//             EditorGUILayout.PropertyField(m_Mute);
//             if (AudioUtil.canUseSpatializerEffect)
//             {
//                 EditorGUILayout.PropertyField(m_Spatialize, Styles.spatializeLabel);
//                 using (new EditorGUI.DisabledScope(!m_Spatialize.boolValue))
//                 {
//                     EditorGUILayout.PropertyField(m_SpatializePostEffects, Styles.spatializePostEffectsLabel);
//                 }
//             }
//             EditorGUILayout.PropertyField(m_BypassEffects);
//             if (targets.Any(t => (t as AudioSource).outputAudioMixerGroup != null))
//             {
//                 using (new EditorGUI.DisabledScope(true))
//                 {
//                     EditorGUILayout.PropertyField(m_BypassListenerEffects);
//                 }
//             }
//             else
//             {
//                 EditorGUILayout.PropertyField(m_BypassListenerEffects);
//             }
//             EditorGUILayout.PropertyField(m_BypassReverbZones);

//             EditorGUILayout.PropertyField(m_PlayOnAwake);
//             EditorGUILayout.PropertyField(m_Loop);

//             EditorGUILayout.Space();
//             EditorGUIUtility.sliderLabels.SetLabels(Styles.priorityLeftLabel, Styles.priorityRightLabel);
//             EditorGUILayout.IntSlider(m_Priority, 0, 256, Styles.priorityLabel);
//             EditorGUIUtility.sliderLabels.SetLabels(null, null);
//             EditorGUILayout.Space();
//             EditorGUILayout.Slider(m_Volume, 0f, 1.0f, Styles.volumeLabel);
//             EditorGUILayout.Space();
//             EditorGUILayout.Slider(m_Pitch, -3.0f, 3.0f, Styles.pitchLabel);

//             EditorGUILayout.Space();

//             EditorGUIUtility.sliderLabels.SetLabels(Styles.panLeftLabel, Styles.panRightLabel);
//             EditorGUILayout.Slider(m_Pan2D, -1f, 1f, Styles.panStereoLabel);
//             EditorGUIUtility.sliderLabels.SetLabels(null, null);
//             EditorGUILayout.Space();

//             // 3D Level control
//             EditorGUIUtility.sliderLabels.SetLabels(Styles.spatialLeftLabel, Styles.spatialRightLabel);
//             AnimProp(Styles.spatialBlendLabel, m_AudioCurves[kSpatialBlendCurveID].curveProp, 0.0f, 1.0f, false);
//             EditorGUIUtility.sliderLabels.SetLabels(null, null);
//             EditorGUILayout.Space();

//             // 3D Level control
//             AnimProp(Styles.reverbZoneMixLabel, m_AudioCurves[kReverbZoneMixCurveID].curveProp, 0.0f, 1.1f, false);
//             EditorGUILayout.Space();

//             m_Expanded3D = EditorGUILayout.Foldout(m_Expanded3D, "3D Sound Settings", true);
//             if (m_Expanded3D)
//             {
//                 EditorGUI.indentLevel++;
//                 Audio3DGUI();
//                 EditorGUI.indentLevel--;
//             }

//             serializedObject.ApplyModifiedProperties();
//             if (m_LowpassObject != null)
//                 m_LowpassObject.ApplyModifiedProperties();
//         }

//         private static void SetRolloffToTarget(SerializedProperty property, Object target)
//         {
//             property.SetToValueOfTarget(target);
//             property.serializedObject.FindProperty("rolloffMode").SetToValueOfTarget(target);
//             property.serializedObject.ApplyModifiedProperties();
//             EditorUtility.ForceReloadInspectors();
//         }

//         private void Audio3DGUI()
//         {
//             EditorGUILayout.Slider(m_DopplerLevel, 0.0f, 5.0f, Styles.dopplerLevelLabel);

//             // Spread control
//             AnimProp(Styles.spreadLabel, m_AudioCurves[kSpreadCurveID].curveProp, 0.0f, 360.0f, true);

//             // Rolloff mode
//             if (m_RolloffMode.hasMultipleDifferentValues ||
//                 (m_RolloffMode.enumValueIndex == (int)AudioRolloffMode.Custom && m_AudioCurves[kRolloffCurveID].curveProp.hasMultipleDifferentValues)
//             )
//             {
//                 EditorGUILayout.TargetChoiceField(m_AudioCurves[kRolloffCurveID].curveProp, Styles.rolloffLabel , SetRolloffToTarget);
//             }
//             else
//             {
//                 EditorGUILayout.PropertyField(m_RolloffMode, Styles.rolloffLabel);

//                 if ((AudioRolloffMode)m_RolloffMode.enumValueIndex != AudioRolloffMode.Custom)
//                 {
//                     EditorGUI.BeginChangeCheck();
//                     EditorGUILayout.PropertyField(m_MinDistance);
//                     if (EditorGUI.EndChangeCheck())
//                     {
//                         m_MinDistance.floatValue = Mathf.Clamp(m_MinDistance.floatValue, 0, m_MaxDistance.floatValue / 1.01f);
//                     }
//                 }
//                 else
//                 {
//                     using (new EditorGUI.DisabledScope(true))
//                     {
//                         EditorGUILayout.LabelField(m_MinDistance.displayName, Styles.controlledByCurveLabel);
//                     }
//                 }
//             }

//             // Max distance control
//             EditorGUI.BeginChangeCheck();
//             EditorGUILayout.PropertyField(m_MaxDistance);
//             if (EditorGUI.EndChangeCheck())
//                 m_MaxDistance.floatValue = Mathf.Min(Mathf.Max(Mathf.Max(m_MaxDistance.floatValue, 0.01f), m_MinDistance.floatValue * 1.01f), 1000000.0f);

//             Rect r = GUILayoutUtility.GetAspectRect(1.333f, GUI.skin.textField);
//             r.xMin += EditorGUI.indent;
//             if (Event.current.type != EventType.Layout && Event.current.type != EventType.Used)
//             {
//                 m_CurveEditor.rect = new Rect(r.x, r.y, r.width, r.height);
//             }

//             // Draw Curve Editor
//             UpdateWrappersAndLegend();
//             GUI.Label(m_CurveEditor.drawRect, GUIContent.none, "TextField");

//             m_CurveEditor.hRangeLocked = Event.current.shift;
//             m_CurveEditor.vRangeLocked = EditorGUI.actionKey;

//             m_CurveEditor.OnGUI();

//             // Draw current listener position
//             if (targets.Length == 1)
//             {
//                 AudioSource t = (AudioSource)target;
//                 AudioListener audioListener = (AudioListener)FindObjectOfType(typeof(AudioListener));
//                 if (audioListener != null)
//                 {
//                     float distToListener = (AudioUtil.GetListenerPos() - t.transform.position).magnitude;
//                     DrawLabel("Listener", distToListener, r);
//                 }
//             }

//             // Draw legend
//             DrawLegend();

//             if (!m_CurveEditor.InLiveEdit())
//             {
//                 // Check if any of the curves changed
//                 foreach (AudioCurveWrapper audioCurve in m_AudioCurves)
//                 {
//                     if ((m_CurveEditor.GetCurveWrapperFromID(audioCurve.id) != null) && (m_CurveEditor.GetCurveWrapperFromID(audioCurve.id).changed))
//                     {
//                         AnimationCurve changedCurve = m_CurveEditor.GetCurveWrapperFromID(audioCurve.id).curve;

//                         // Never save a curve with no keys
//                         if (changedCurve.length > 0)
//                         {
//                             audioCurve.curveProp.animationCurveValue = changedCurve;
//                             m_CurveEditor.GetCurveWrapperFromID(audioCurve.id).changed = false;

//                             // Volume curve special handling
//                             if (audioCurve.type == AudioCurveType.Volume)
//                                 m_RolloffMode.enumValueIndex = (int)AudioRolloffMode.Custom;
//                         }
//                     }
//                 }
//             }
//         }

//         void UpdateWrappersAndLegend()
//         {
//             if (m_CurveEditor.InLiveEdit())
//                 return;

//             // prevent rebuilding wrappers if any curve has changes
//             if (m_CurveEditor.animationCurves != null)
//             {
//                 for (int i = 0; i < m_CurveEditor.animationCurves.Length; i++)
//                 {
//                     if (m_CurveEditor.animationCurves[i].changed)
//                         return;
//                 }
//             }

//             m_CurveEditor.animationCurves = GetCurveWrapperArray();
//             SyncShownCurvesToLegend(GetShownAudioCurves());
//         }

//         void DrawLegend()
//         {
//             List<Rect> legendRects = new List<Rect>();
//             List<AudioCurveWrapper> curves = GetShownAudioCurves();

//             Rect legendRect = GUILayoutUtility.GetRect(10, 20);
//             legendRect.x += 4 + EditorGUI.indent;
//             legendRect.width -= 8 + EditorGUI.indent;
//             int width = Mathf.Min(75, Mathf.FloorToInt(legendRect.width / curves.Count));
//             for (int i = 0; i < curves.Count; i++)
//             {
//                 legendRects.Add(new Rect(legendRect.x + width * i, legendRect.y, width, legendRect.height));
//             }

//             bool resetSelections = false;
//             if (curves.Count != m_SelectedCurves.Length)
//             {
//                 m_SelectedCurves = new bool[curves.Count];
//                 resetSelections = true;
//             }

//             if (EditorGUIExt.DragSelection(legendRects.ToArray(), ref m_SelectedCurves, GUIStyle.none) || resetSelections)
//             {
//                 // If none are selected, select all
//                 bool someSelected = false;
//                 for (int i = 0; i < curves.Count; i++)
//                 {
//                     if (m_SelectedCurves[i])
//                         someSelected = true;
//                 }
//                 if (!someSelected)
//                 {
//                     for (int i = 0; i < curves.Count; i++)
//                     {
//                         m_SelectedCurves[i] = true;
//                     }
//                 }

//                 SyncShownCurvesToLegend(curves);
//             }

//             for (int i = 0; i < curves.Count; i++)
//             {
//                 EditorGUI.DrawLegend(legendRects[i], curves[i].color, curves[i].legend.text, m_SelectedCurves[i]);
//                 if (curves[i].curveProp.hasMultipleDifferentValues)
//                 {
//                     GUI.Button(new Rect(legendRects[i].x, legendRects[i].y + 20, legendRects[i].width, 20), "Different");
//                 }
//             }
//         }

//         private List<AudioCurveWrapper> GetShownAudioCurves()
//         {
//             return m_AudioCurves.Where(f => m_CurveEditor.GetCurveWrapperFromID(f.id) != null).ToList();
//         }

//         private void SyncShownCurvesToLegend(List<AudioCurveWrapper> curves)
//         {
//             if (curves.Count != m_SelectedCurves.Length)
//                 return; // Selected curves in sync'ed later in this frame

//             for (int i = 0; i < curves.Count; i++)
//                 m_CurveEditor.GetCurveWrapperFromID(curves[i].id).hidden = !m_SelectedCurves[i];

//             // Need to apply animation curves again to synch selections
//             m_CurveEditor.animationCurves = m_CurveEditor.animationCurves;
//         }

//         void DrawLabel(string label, float value, Rect r)
//         {
//             Vector2 size = Styles.labelStyle.CalcSize(new GUIContent(label));
//             size.x += 2;
//             Vector2 posA = m_CurveEditor.DrawingToViewTransformPoint(new Vector2(value / m_MaxDistance.floatValue, 0));
//             Vector2 posB = m_CurveEditor.DrawingToViewTransformPoint(new Vector2(value / m_MaxDistance.floatValue, 1));
//             GUI.BeginGroup(r);
//             Color temp = Handles.color;
//             Handles.color = new Color(1, 0, 0, 0.3f);
//             Handles.DrawLine(new Vector3(posA.x  , posA.y, 0), new Vector3(posB.x  , posB.y, 0));
//             Handles.DrawLine(new Vector3(posA.x + 1, posA.y, 0), new Vector3(posB.x + 1, posB.y, 0));
//             Handles.color = temp;
//             GUI.Label(new Rect(Mathf.Floor(posB.x - size.x / 2), 2, size.x, 15), label, Styles.labelStyle);
//             GUI.EndGroup();
//         }

//         internal static void AnimProp(GUIContent label, SerializedProperty prop, float min, float max, bool useNormalizedValue)
//         {
//             if (prop.hasMultipleDifferentValues)
//             {
//                 EditorGUILayout.TargetChoiceField(prop, label);
//                 return;
//             }

//             AnimationCurve curve = prop.animationCurveValue;
//             if (curve == null)
//             {
//                 Debug.LogError(label.text + " curve is null!");
//                 return;
//             }
//             else if (curve.length == 0)
//             {
//                 Debug.LogError(label.text + " curve has no keys!");
//                 return;
//             }

//             Rect position = EditorGUILayout.GetControlRect();
//             EditorGUI.BeginProperty(position, label, prop);
//             if (curve.length != 1)
//             {
//                 using (new EditorGUI.DisabledScope(true))
//                 {
//                     EditorGUI.LabelField(position, label.text, Styles.controlledByCurveLabel);
//                 }
//             }
//             else
//             {
//                 float f = useNormalizedValue ? Mathf.Lerp(min, max, curve.keys[0].value) : curve.keys[0].value;
//                 f = MathUtils.DiscardLeastSignificantDecimal(f);
//                 EditorGUI.BeginChangeCheck();
//                 if (max > min)
//                     f = EditorGUI.Slider(position, label, f, min, max);
//                 else
//                     f = EditorGUI.Slider(position, label, f, max, min);

//                 if (EditorGUI.EndChangeCheck())
//                 {
//                     Keyframe kf = curve.keys[0];
//                     kf.time = 0.0f;
//                     kf.value = useNormalizedValue ? Mathf.InverseLerp(min, max, f) : f;
//                     curve.MoveKey(0, kf);
//                 }
//             }
//             EditorGUI.EndProperty();

//             prop.animationCurveValue = curve;
//         }

//         void OnSceneGUI()
//         {
//             if (!target)
//                 return;
//             AudioSource source = (AudioSource)target;

//             Color tempColor = Handles.color;
//             if (source.enabled)
//                 Handles.color = new Color(0.50f, 0.70f, 1.00f, 0.5f);
//             else
//                 Handles.color = new Color(0.30f, 0.40f, 0.60f, 0.5f);

//             Vector3 position = source.transform.position;

//             EditorGUI.BeginChangeCheck();
//             float minDistance = Handles.RadiusHandle(Quaternion.identity, position, source.minDistance, true);
//             float maxDistance = Handles.RadiusHandle(Quaternion.identity, position, source.maxDistance, true);
//             if (EditorGUI.EndChangeCheck())
//             {
//                 Undo.RecordObject(source, "AudioSource Distance");
//                 source.minDistance = minDistance;
//                 source.maxDistance = maxDistance;
//             }

//             Handles.color = tempColor;
//         }
//     }
// }