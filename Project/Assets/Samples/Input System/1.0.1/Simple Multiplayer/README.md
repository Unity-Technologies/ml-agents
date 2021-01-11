# Simple Multiplayer Demo

This demo shows a simple local multiplayer setup. Players can join by pressing buttons on the supported devices. As players join, the screen is subdivided in split-screen fashion.

Joining is handled by the `PlayerManager` GameObject in the scene which has the `PlayerInputManager` component added to it. The component references [`Player.prefab`](./Player.prefab) which is instantiated for each player that joins the game.

The prefab contains a GameObject that has a `PlayerInput` component added to it. The component references the [actions](./SimpleMultiplayerControls.inputactions) available to each player which, by means of the control schemes defined in the asset, also determine the devices (and combinations of devices) supported by the game.

The actions available to each player are intentionally kept simple for this demonstration in order to not add irrelevant details. The only action available to players is `Teleport` which players can trigger through a button on their device. When trigger, they will be teleported to a random position within the game area. This serves to demonstrate that player inputs are indeed separate.

Note that each `PlayerInput` also references a `Camera` which is specific to each player. This is used by `PlayerInputManager` to configure the split-screen setup.
