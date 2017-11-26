SteamVR plugin for Unity - v1.2.2
Copyright (c) Valve Corporation, All rights reserved.


Quickstart:

To use, simply add the SteamVR_Camera script to your Camera object(s).  Everything else gets set up at
runtime.  See the included quickstart guide for more details.


Requirements:

The SteamVR runtime must be installed.  This can be found in Steam under Tools.


Changes for v1.2.2:

* Updated to SteamVR runtime v1497390325 and SDK version 1.0.8.

* [General] Switched caching SteamVR_Events.Actions from Awake to constructors to fix hot-loading of scripts in the Editor.

* [General] Switched remaining coroutines away from using strings (to avoid issues with obfuscators).

* [General] Switched from using deprecated Transform.FindChild to Transform.Find.

* [General] Added #if !UNITY_METRO where required to allow compiling for UWP.

* [UpdatePoses] Switched to using static delegates (Camera.onPreCull or Application.onBeforeRender depending on version) for updating poses.

* [UpdatePoses] Deprecated SteamVR_UpdatePoses component.

* [MixedReality] Added rgba settings to externalcamera.cfg for overriding foreground chroma key (default 0,0,0,0).

* [MixedReality] Exposed SteamVR_ExternalCamera.Config settings in Unity Editor inspector for easy tweaking.

* [MixedReality] Added file watcher to externalcamera.cfg to allow real-time editing.

* [MixedReality] Fixed antialiasing complaint in Unity 5.6+.

* [MixedReality] Added second pass to foreground camera when using PostProcessingBehaviour since those fx screw up the alpha channel.

* [ControllerManager] Added code to protect against double-hiding of controllers.

* [InteractionSystem] Sub-objects now inherit layer and tag of spawning object (ControllerButtonHints, ControllerHoverHighlight, Hand, SpawnRenderModel).


Changes for v1.2.1:

* Updated to SteamVR runtime v1485823399 and SDK version 1.0.6.

* Switched SteamVR_Events.SystemAction from using strings to specify event type over to their associated enum values.

* Fixed an issue with using WWW in static constructors.

* Added Unity Preferences for SteamVR to allow disabling automatic enabling of native OpenVR support in Unity 5.4 or newer.
https://github.com/ValveSoftware/steamvr_unity_plugin/issues/8
https://github.com/ValveSoftware/steamvr_unity_plugin/pull/9

* Added UNITY_SHADER_NO_UPGRADE to all shaders to avoid log spam in later versions of Unity for issues that have already been fixed but the compiler isn't able to detect.

* Specified Vulkan support for Interaction System shaders.

* Fix for crash in Interaction_Example selecting BowPickup:
https://github.com/ValveSoftware/steamvr_unity_plugin/issues/4

* Cleaned up unused fields:
https://github.com/ValveSoftware/steamvr_unity_plugin/issues/2

* Updated Interaction System's LinearDrive to initialize using linearMapping.value.
https://github.com/ValveSoftware/steamvr_unity_plugin/pull/5

* Updated Interaction System documetation to fix a few errors.

* Added an icon for all Interaction System scripts.

* Fixes for SteamVR on Linux.


Changes for v1.2.0:

* Updated to SteamVR runtime v1481926580 and SDK version 1.0.5.

* Replaced SteamVR_Utils.Event with SteamVR_Events.<EventName> to avoid runtime memory allocation associated with use of params object[] args.

* Added SteamVR_Events.<EventName>Action to make it easy to wrap callbacks to avoid memory allocation when components are frequently enabled/disabled at runtime.

* Fixed other miscellaneous runtime memory allocation in SteamVR_Render and SteamVR_RenderModels.  (Suggestions by unity3d user @8bitgoose.)

* Integrated fix for SteamVR_LaserPointer direction (from github user @fredsa).

* Integrated fixes and comments for SteamVR_Teleporter (from github user @natewinck).

* Removed SteamVR_Status and SteamVR_StatusText as they were using SteamVR_Utils.Event with generic strings which is no longer allowed.

* Added SteamVR_Controller.assignAllBeforeIdentified (to allow controller to be assigned before identified as left vs right).  Suggested by github user @chrwoizi.

* Added SteamVR_Controller.UpdateTargets public interface.  This allows spawning the component at runtime.  Suggested by github user @demonixis.

* Fixed bug with SteamVR_TrackedObject when specifying origin.  Suggested by github user @fredsa.

* Fixed issue with head camera reference in SteamVR_Camera.  Suggested by github user @pedrofe.

Known issues:

* The current beta version of Unity 5.6 breaks the normal operation of the SteamVR_UpdatePoses component (required for tracked controllers).
To work around this in the meantime, you will need to manually add the SteamVR_UpdatePoses component to your main camera.


Changes for v1.1.1:

* Updated to SteamVR runtime v1467410709 and SDK version 1.0.2.

* Updated Copyright notice.

* Added SteamVR_TrackedCamera for accessing tracked camera video stream and poses.

* Added SteamVR_TestTrackedCamera scene and associated script to demonstrate how to use SteamVR_TrackedCamera.

* Fix for SteamVR_Fade shader to account for changes in Unity 5.4.

* SteamVR_GameView will now use the compositor's mirror texture to render the companion window (pre-Unity 5.4 only).

* Renamed SteamVR_LoadLevel 'externalApp' to 'internalProcess' to reflect actual functionality.

* Fixed issue with SteamVR_PlayArea material loading due to changes in Unity 5.4.

* Added Screenshot support handling for stereo panoramas generation.

* Removed code that was setting Time.maximumDeltaTime as this was causing issues.


Changes for v1.1.0:

* Fix for error building standalone in SteamVR_LoadLevel.

* Set SteamVR_TrackedObject.isValid to false when disabled.


Changes for v1.0.9:

* Updated to SteamVR runtime v1461626459 and SDK version 0.9.20.

* Updated workshop texture used in sea of cubes example level to use web page from SteamVR (was previously from Portal).

* Updated various SDK changes to Unity in 5.4 betas.

* Added controllerModeState to RenderModel component to control additional features like scrollwheel visibility.

* RenderModels now respond to model skin changes.

* Removed OnGUI and associated help text (i.e. "You may now put on your headset." notification) as this was causing unnecessary overhead.

* Fix to SteamVR_Render not turning back on if all cameras were disabled and then re-enabled.

* Hooked up SteamVR_Render.pauseRendering in Unity 5.4 native OpenVR integration.

* Fix for input_focus event sometimes getting sent inappropriately.

* Fix for timeScale handling.

* Fix for SteamVR_PlayArea not finding its material in editor in Unity 5.4 due to changes in how Unity handles asset loading.

* Miscellaneous fixes to reduce hitching when using SteamVR_LoadLevel to handle scene transitions.

* Hooked up SteamVR_Camera.sceneResolutionScale to Unity 5.4's native vr integration render target scaling.

* Forced SteamVR initialization check in SteamVR_Camera.enable (and bail upon failure) in Unity 5.4 (was already doing this in older builds).

* Better handling of SteamVR_Ears component with old content.

* Keep legacy head object around in case external components were referencing it (was previously getting deleted in Unity 5.4 as the head motion is now applied to the "eyes" object).

* Miscellaneous fixes for SteamVR_TrackedController and SteamVR_Teleporter.

* Fixed up Extra scenes SteamVR_TestThrow and SteamVR_TestIK.

* Added stereo panorama screenshot support to SteamVR_Skybox.

* Removed use of deprecated UnityEditorInternal.VR.VREditor.InitializeVRPlayerSettingsForBuildTarget(BuildTargetGroup.Standalone);


Changes for v1.0.8:

* Updated to SteamVR runtime v1457155403.

* Updated to work with native OpenVR integration introduced in Unity 5.4.  In this and newer versions, openvr_api.dll will be automatically deleted when launching since it ships as part of Unity now.

* C# interop now exports arrays as individual elements to avoid the associated memory allocation overhead passing data back and forth between native and managed code.

* Applications should no longer call GetGenericInterface directly, but instead use the accessors provided by Valve.VR.OpenVR (e.g. OpenVR.System for the IVRSystem interface).

* Added SteamVR_ExternalCamera for filming mixed reality videos.  Automatically enabled when externalcamera.cfg is added to the root of your projects (next to Assets or executable), and toggled by the presence of a third controller.

* Render models updated to load asynchronously.  Sends "render_model_loaded" event when finished.

* Added 'shader' property to render models to allow using alternate shaders. This also creates a dependency within the scene to ensure the shader is loaded in baked builds.

* Fix for render model components not respecting render model scale.

* SteamVR_Render.lockPhysicsUpdateRateToRenderFrequency now respects Time.timeScale.

* SteamVR_LoadLevel now hides overlays when finished to avoid persisting performance degredation.

* Added ability to launch external applications via SteamVR_LoadLevel.

* Added option to load levels non-asynchronously in SteamVR_LoadLevel since Unity crashes on some content when using asyn loading.

* SteamVR auto-disabled if initialization fails, to avoid continual retries.

* Updated SteamVR_ControllerManager to get controller indices from the runtime (via IVRSystem.GetTrackedDeviceIndexForControllerRole).

* SteamVR_ControllerManager now allows you to assign additional controllers to game objects.

* [CameraRig] prefab now listens for a third controller connection which will enable mixed reality recording mode in the game view.

* AudioListener is now transferred to a child of the eye camera called "ears" to allow controlling rotation independently when using speakers instead of headphones.

* Flare Layer is no longer transferred from eye camera to game view camera.


Changes for v1.0.7:

* Updated to SteamVR runtime v1448479831.

* Many enums updated to reflect latest SDK cleanup (v0.9.12).

* Various fixes to support color space changes in the SDK.

* Render models set the layer on their child components now to match their own.

* Added a bool 'Load Additive' to SteamVR_LoadLevel script to optionally load the level additively, as well as an optional 'Post Load Settle Time'.

* Fixed some issues with SteamVR_LoadLevel fading to a color with 'Show Grid' set to false.

* Fixed an issue with orienting the loading screen in the SteamVR_LoadLevel script when using 'Loading Screen Distance'.


Changes for v1.0.6:

* Updated to SteamVR runtime v1446847085.

* Added SteamVR_LevelLoad script to help smooth level transitions.

* Added 'Take Snapshot' button to SteamVR_Skybox to automate creation of cubemap assets.

* SteamVR_RenderModel now optionally creates subcomponents for buttons, etc. and optionally updates them dynamically to reflect pulling trigger, etc.

* Added SteamVR_TestIK scene to Extras.

* Added SteamVR.enabled which can be set to false to keep SteamVR.instance from initializing SteamVR.


Changes for v1.0.5:

* Updated to SteamVR runtime build #826021 (v.1445485596).

* Removed TrackedDevices from [CameraRig] prefab (these were only ever meant to be in the example scene.

* Added support for new native plugin interface.

* Enabled MSAA in OpenGL as that appears to be fixed in the latest version of Unity.

* Fix for upside-down rendering in OpenGL.

* Moved calls to IVRCompositor::WaitGetPoses and Submit to Unity's render thread.

* Couple fixes to prevent SteamVR from getting re-initialized when stopping the Editor preview.

* Fix for hitches caused by SteamVR_PlayArea when not running SteamVR.


Changes for v1.0.4:

* Updated to SteamVR runtime build #768489 (v.1441831863).

* Added SteamVR_Skybox for setting a cubemap in the compositor (useful for scene transitions).

* Fix for RenderModels disappearing across scene transitions, and disabling use of modelOverride at runtime.

* Added lockPhysicsUpdateRateToRenderFrequency to SteamVR_Render ([SteamVR] prefab) for apps that want to run their physics sim at a lower frequency.  Locked (true) by default.

* Made per-eye culling masks easier to use.  (See http://steamcommunity.com/app/250820/discussions/0/535152276589455019/)

* Exposed min/max curve distance settings for high quality overlay.  Note: High quality overlay not currently supported in Rift Direct Mode and falls back to normal (flat-only) overlay render path.

* Added 'valid' property to SteamVR_Controller.  This is useful for detecting the controller is plugged in before tracking comes online.


Changes for v1.0.3:

* Updated to SteamVR runtime build #710329 (v.1438035413).

* Added SteamVR_Controller.DeviceRelation.FarthestLeft/Right for GetDeviceIndex helper function.
Note: You can also use SteamVR.instance.hmd.GetSortedTrackedDeviceIndicesOfClass.

* Updated and fixed SteamVR_Controller.GetDeviceIndex to act more like people expect.

* Fix for SteamVR_Controller.angularVelocity (velocity reporting has also been fixed in the runtime).

* Renamed SteamVR_Controller.valid to hasTracking

* Removed SteamVR_Overlay visibility, systemOverlayVisible and activeSystemOverlay properties.

* Added collection of handy scripts to Assets/SteamVR/Extras: GazeTracker, IK (simple two-bone),
LaserPointer, Teleporter, TestThrow (with example scene) and TrackedController.

* Fix for hidden area mesh render order.

* Fix for render models not showing up after playing scene once in editor.

* Added controller manager left and right nodes to camera rig.  These are automatically disabled while the
dashboard is visible to avoid conflict with the dashboard rendering controllers.  If you are handling tracked
controllers using another method, you are encouraged to implement something similar using the input_focus event.

* OpenVR runtime events are now broadcast via the SteamVR_Utils.Event system.  The events can be found here:
https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h and are broadcast in Unity with their
prefix "VREvent_" stripped off.

* Added handling of dashboard visibility and quit events.

* Added SteamVR_Render.pauseGameWhenDashboardIsVisible (defaults to true).

* Allow Unity to buffer up frames for its companion window to avoid any latency introduction

* Lock physics update rate (Time.fixedDeltaTime) to match render frequency.

* SteamVR_Camera (i.e. 'eye' objects) are moved back to the 'head' location when not rendering.

* Simplified SteamVR_Camera Expand/Collapse functionality (now uses existing parent as origin if available).

* Added SteamVR_PlayArea component to visualize different size spaces to target.

* Exposed SteamVR_Overlay.curvedRange for the high-quality curved overlay render path.


Changes for v1.0.2:

* Updated to SteamVR runtime build #655277.

* Added check for new version and prompt to download.

* Moved remaining in-code shaders to separate shader assets.

* Switched RenderModels back to using Standard shader (to avoid having to manually add Unlit to the always load
assets list).

* RenderModels now provides a drop down list populated with available render models to preview.  This is useful
for displaying various controller models in Editor to line up attachments appropriately.

* Fix for [SteamVR] instance sometimes showing up twice in a scene between level loads and stomping existing
settings.

* Switched Overlay over to using new interface.  Please report any functional differences to the SteamVR forums.

* Added button in example escape menu [Status] to easily switch between Standing and Seated tracking space.

* Miscellaneous color space fixes due to changes in Unity 5.1 rendering.

* Added drawOverlay bool to GameView component to disable rendering the overlay texture automatically on top.

* Eye offsets now get updated at runtime to react to any dynanamic IPD changes.

* Added "hair-trigger" support to SteamVR_Controller.


Changes for v1.0.1:

* Updated to SteamVR runtime build #629708.

* Added accessors to SteamVR_Controller for working with input.

* Added TestController script for verifying controller functionality.

* Added CameraFlip to compensate for Unity's quirk of rendering upsidedown on Windows (was previously
corrected for in the compositor).

* Removed use of UNITY_5_0 defines as this was causing problems with newer versions of Unity 5.

* Shared render target size is now slightly larger to account for overlapping fovs.

* Fix for gamma issues with deferred rendering and hdr.

Note: MSAA is really important for rendering in VR, however, Unity does not support MSAA in deferred rendering.
It is therefore recommended that you use Unity's Forward rendering path.  Unity's Forward rendering path also
does not support MSAA when using HDR render buffers.  If you need to disable MSAA, you should at least attempt
to compensate with an AA post fx.  The MSAA settings for SteamVR's render buffers are controlled via Unity's
Quality settings (Edit > Project Settings > Quality > Anti Aliasing).


Upgrading from previous versions:

The easiest and safest thing to do is to delete your SteamVR folder, and any files and folders in your
Plugins directory called 'openvr_api', 'steam_api' or 'steam_unity' (and variants).  Additionally, verify there
are no SteamVR files found in Assets/Editor.  Then import the new unitypackage into your project.

This latest version has been greatly simplified.  SteamVR_CameraEye has been removed as well as the menu
option from SteamVR_Setup to 'Setup Selected Camera(s)'.  The SteamVR_Camera object is instead rendered twice
(once per eye) and the game view rendering handled in SteamVR_GameView.  SteamVR_Camera now has 'head' and
'origin' properties for accessing the associated Transforms, and 'offset' has been deprecated in favor of using
'head'.  By pressing the 'Expand' button below the SteamVR logo in SteamVR_Camera's Inspector, these objects are
automatically created.  This is useful for attaching objects appropriately, and removes the need for managing
separate FollowHead and FollowEyes arrays. Similarly, the RenderComponents list is no longer needed as the
SteamVR_Camera is itself used to render each eye.  And finally, the button below the SteamVR logo will change to
'Collapse' to restore the camera to its previous setup.

SteamVR_Camera's Overlay support has been broken out into a separate SteamVR_Overlay component.  This can be
added to any object in your scene.  If you wish to use it in some scenes, but not others, it is good practice
to add the component to each of your scenes and ensure its Texture is set to None in those that you do not wish
it rendered in.

The experimental binaural audio support has been removed as there are better plugins on the Unity Asset Store now,
and this was an incomplete and unsupported solution.


Files:

Assets/Plugins/openvr_api.cs - This direct wrapper for the native SteamVR SDK support mirrors SteamVR.h and  
is the only script required.  It exposes all functionality provided by SteamVR.  It is not recommended you make  
changes to this file.  It should be kept in sync with the associated openvr_api dll.

The remaining files found in Assets/SteamVR/Scripts are provided as a reference implementation, and to get you  
up and running quickly and easily.  You are encouraged to modify these to suit your project's unique needs,  
and provide feedback at http://steamcommunity.com/app/250820 or http://steamcommunity.com/app/358720/discussions

Assets/SteamVR/Scenes/example.unity - A sample scene demonstrating the functionality provided by this plugin.   
This also shows you how to set up a separate camera for rendering gui elements.


Details:

Note that these scripts are a work in progress. Many of these will change in future releases and we will not
necessarily be able to maintain compatibility with this version.

Assets/SteamVR/Scripts/SteamVR.cs - Handles initialization and shutdown of subsystems.  Use SteamVR.instance
to access.  This may return null if initialization fails for any reason.  Use SteamVR.active to determine if
VR has been initialized without attempting to initialized it in the process.

Assets/SteamVR/Scripts/SteamVR_Camera.cs - Adds VR support to your existing camera object.

To combat stretching incurred by distortion correction, we render scenes at a higher resolution off-screen.
Since all camera's in Unity are rendered sequentially, we share a single static render texture across each
eye camera.  SteamVR provides a recommended render target size as a minimum to account for distortion,
however, rendering to a higher resolution provides additional multisampling benefits at the associated
expense.  This can be controlled via SteamVR_Camera.sceneResolutionScale.

Note: Both GUILayer and FlareLayer are not compatible with SteamVR_Camera since they render in screen space
rather than world space. These are automatically moved the SteamVR_GameView object which itself is automatically
added to the SteamVR_Camera's parent 'head' object.  The AudioListener also gets transferred to the head in order
for audio to be properly spacialized.

Assets/SteamVR/Scripts/SteamVR_Overlay.cs - This component is provided to assist in rendering 2D content in VR.
The specified texture is composited into the scene on a virtual curved surface using a special render path for
increased fidelity.  See the [Status] prefab in the example scene for how to set this up.  Since it uses GUIText,
it should be dragged into the Hierarchy window rather than into the Scene window so it retains its default position
at the origin.

Assets/SteamVR/Scripts/SteamVR_TrackedObject.cs - Add this to any object that you want to use tracking.  The
hmd has one set up for it automatically.  For controllers, select the index of the object to map to.  In general
you should parent these objects to the camera's 'origin' object so they track in the same space.  However, if
that is inconvenient, you can specify the 'origin' in the TrackedObject itself.

Assets/SteamVR/Scripts/SteamVR_RenderModel.cs - Dynamically creates associated SteamVR provided models for tracked
objects.  See <SteamVR Runtime Path>/resources/rendermodels for the full list of overrides.

Assets/SteamVR/Scripts/SteamVR_Utils.cs - Various bits for working with the SteamVR API in Unity including a  
simple event system, a RigidTransform class for working with vector/quaternion pairs, matrix conversions, and  
other useful functions.


Prefabs:

[CameraRig] - This is the camera setup used by the example scene.  It is simply a default camera with the
SteamVR_Camera component added to it, and the Expand button clicked.  It also includes a full set of Tracked Devices
which will display and follow any connected tracked devices (e.g. controllers, base stations and cameras).

[Status] - The prefab is for demonstration purposes only.  It adds an escape menu to your scene.
Note: It uses the SteamVR_Overlay component, which is rather expensive rendering-wise.

[SteamVR] - This object controls some global settings for SteamVR, most notably Tracking Space.  Legacy projects
that want their viewed automatically centered on startup if not configured or to use the seated calibrated position
should switch Tracking Space to Seated.  This object is created automatically on startup if not added and defaults
to Standing Tracking Space.  It also provides the ability to set special masks for rendering each eye (in case you
want to do something differently per-eye) and some simple help text that demonstrates rendering only to the
companion window (which can be cleared or customized here).


GUILayer, GUIText, and GUITexture:

The recommended way for drawing 2D content is through SteamVR_Overlay.  There is an example of how to set this up
in the example scene.  GUIText and GUITexture use their Transform to determine where they are drawn, so these
objects will need to live near the origin.  You will need to set up a separate camera using a Target Texture.  To
keep it from rendering other elements of your scene, you should create a unique layer used by all of your gui
elements, and set the camera's Culling Mask to only draw those items.  Set its depth to -1 to ensure it gets
updated before composited into the final view.


OnGUI:

Assets/SteamVR/Scripts/SteamVR_Menu.cs demonstrates use of OnGUI with SteamVR_Camera's overlay texture.  The  
key is to set RenderTexture.active and restore it afterward.  Beware when also using a camera to render to the  
same texture as it may clear your content.


Camera layering:

One powerful feature of Unity is its ability to layer cameras to render scenes (e.g. drawing a skybox scene
with one camera, the rest of the environment with a second, and maybe a third for a 3D hud).  This is performed
by setting the latter cameras to only clear the depth buffer, and leveraging the cameras' cullingMask to control
which items get rendered per-camera, and depth to control order.


Camera scale:

Setting SteamVR_Camera's gameObject scale will result in the world appearing (inversely) larger or smaller.
This can be used to powerful effect, and is useful for allowing you to build skybox geometry at a sane scale
while still making it feel far away.  Similarly, it allows you to build geometry at scales the physics engine
and nav mesh generation prefers, while visually feeling much smaller or larger.  Of course, if you are building
geometry to real-world scale you should leave this at its default of 1,1,1.  Once a SteamVR_Camera has been
expanded, its 'origin' Transform should be scaled instead.


Camera masking:

By manually adding a GameObject with the SteamVR_Render component on it to your scene, you can specify a left
and right culling mask to use to control rendering per eye if necessary.


Events:

SteamVR fires off several events.  These can be handled by registering for them through
SteamVR_Events.<EventType>.Listen.  Be sure to remove your handler when no longer needed.
The best pattern is to Listen and Remove in OnEnable and OnDisable respectively.

Initializing - This event is sent when the hmd's tracking status changes to or from Unitialized.

Calibrating - This event is sent when starting or stopping calibration with the new state.

OutOfRange - This event is sent when losing or reacquiring absolute positional tracking.  This will 
never fire for the Rift DK1 since it does not have positional tracking.  For camera based trackers, this 
happens when the hmd exits and enters the camera's view.

DeviceConnected - This event is sent when devices are connected or disconnected.  The device index is passed
as the first argument, and the connected status (true / false) as the second argument.


Keybindings (if using the [Status] prefab):

Escape/Start - toggle menu
PageUp/PageDown - adjust scale
Home - reset scale
I - toggle frame stats on/off


Deploying on Steam:

If you are releasing your game on Steam (i.e. have a Steam ID and are calling Steam_Init through the  
Steamworks SDK), then you may want to check ISteamUtils::IsSteamRunningInVRMode() in order to determine if you  
should automatically launch into VR mode or not.


Known Issues:

* If Unity finds an Assets\Plugins\x86 folder, it will ignore all files in Assets\Plugins.  You will need to
either move openv_api.dll into the x86 subfolder, or move the dlls in the x86 folder up a level and delete
the x86 folder.


Troubleshooting:

* "Failed to connect to vrserver" - This often happens the first time you launch.  Often simply trying a second time
will clear this up.

* HmdError_Init_VRClientDLLNotFound - Make sure the SteamVR runtime is installed.  This can be found in Steam
under Tools.  Try uninstalling and reinstalling SteamVR.  Try deleting <user>/AppData/Local/OpenVR/openvrpaths.vrpath
and relaunching Steam to regenerate this file.

* HmdError_Init_HmdNotFound - SteamVR cannot detect your VR headset, ensure the USB cable is plugged in.
If that doesn't work, try deleting your Steam/config/steamvr.cfg.

* HmdError_Init_InterfaceNotFound - Make sure your SteamVR runtime is up to date.

* HmdError_IPC_ConnectFailed - SteamVR launches a separate process called vrserver.exe which directly talks
to the hardware.  Games communicate to vrserver through vrclient.dll over IPC.  This error is usually due
to the communication pipe between the two having closed.  Use task manager to verify there are no rogue apps
that got stuck trying to shut down.  Often it's just a matter of the connection timing out the first time
due to long load times.  Launching a second time usually resolves this.

* "Not using DXGI 1.1" - Older versions of Unity used DXGI 1.0 which doesn't support functionality the compositor
requires to operate properly.  To fix this, we've added a hook to Steam to force DXGI 1.1.  To enable this hook
set the environement variable ForceDXGICreateFactory1 = 1 and launch the Unity Editor or your standalone builds
via Steam by manually adding them using the "Add Game..." button found in the lower left of the Library tab.

* Core Parking often causes hitching.  The easiest way to disable core parking is to download the tool called
Core Parking Manager, slide the slider to 100% and click Apply.

