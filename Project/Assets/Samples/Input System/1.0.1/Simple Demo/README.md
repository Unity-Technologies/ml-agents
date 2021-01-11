This sample shows how to set up a simple character controller using the input system. As there is more than one way to do it, the sample illustrates several ways. Each demonstration is set up as a separate scene. The basic functionality in all the scenes is the same. You can move and look around and fire projectiles (colored cubes) into the scene. In some scenes, only gamepads are supported but the more involved demonstrations support several different inputs concurrently.

# SimpleDemo_UsingState

[Source](./SimpleController_UsingState.cs)

This starts off at the lowest level by demonstrating how to wire up input by polling input state directly in a `MonoBehaviour.Update` function. For simplicity's sake it only deals with gamepads but the same mechanism works in equivalent ways for other types of input devices (e.g. using `Mouse.current` and `Keyboard.current`).

The key APIs demonstrated here are `Gamepad.current` and `InputControl.ReadValue`.

```CSharp
public class SimpleController_UsingState : MonoBehaviour
{
    //...

    public void Update()
    {
        var gamepad = Gamepad.current;
        if (gamepad == null)
            return;

        var move = Gamepad.leftStick.ReadValue();
        //...
    }
}
```

# SimpleDemo_UsingActions

[Source](./SimpleController_UsingActions.cs)

This moves one level higher and moves input over to "input actions". These are input abstractions that allow you to bind to input sources indirectly.

In this scene, the actions are embedded directly into the character controller component. This allows setting up the bindings for the actions directly in the inspector. To see the actions and their bindings, select the `Player` object in the hierarchy and look at the `SimpleController_UsingActions` component in the inspector.

The key APIs demonstrated here are `InputAction` and its `Enable`/`Disable` methods and its `ReadValue` method.

```CSharp
public class SimpleController_UsingActions : MonoBehaviour
{
    public InputAction moveAction;
    //...

    public void OnEnable()
    {
        moveAction.Enable();
        //...
    }

    public void OnDisable()
    {
        moveAction.Disable();
        //...
    }

    public void Update()
    {
        var move = moveAction.ReadValue<Vector2>();
        //...
    }
}
```

The sample also demonstrates how to use a `Tap` and a `SlowTap` interaction on the fire action to implement a charged shooting mechanism. Note that in this case, we run the firing logic right from within the action using the action's `started`, `performed`, and `canceled` callbacks.

```CSharp
        fireAction.performed +=
            ctx =>
        {
            if (ctx.interaction is SlowTapInteraction)
            {
                StartCoroutine(BurstFire((int)(ctx.duration * burstSpeed)));
            }
            else
            {
                Fire();
            }
            m_Charging = false;
        };
        fireAction.started +=
            ctx =>
        {
            if (ctx.interaction is SlowTapInteraction)
                m_Charging = true;
        };
        fireAction.canceled +=
            ctx =>
        {
            m_Charging = false;
        };
```

# SimpleDemo_UsingActionAsset

[Source](./SimpleController_UsingActionAsset.cs)

As more and more actions are added, it can become quite tedious to manually set up and `Enable` and `Disable` all the actions. We could use an `InputActionMap` in the component like so

```CSharp
public class SimpleController : MonoBehaviour
{
    public InputActionMap actions;

    public void OnEnable()
    {
        actions.Enable();
    }

    public void OnDisable()
    {
        actions.Disable();
    }
}
```

but then we would have to look up all the actions manually in the action map. A simpler approach is to put all our actions in a separate asset and generate a C# wrapper class that automatically performs the lookup for us.

To create such an `.inputactions` asset, right-click in the Project Browser and click `Create >> Input Actions`. To edit the actions, double-click the `.inputactions` asset and a separate window will come up. The asset we use in this example is [SimpleControls.inputactions](SimpleControls.inputactions).

When you select the asset, note that `Generate C# Class` is ticked in the import settings. This triggers the generation of [SimpleControls.cs](SimpleControls.cs) based on the `.inputactions` file.

Regarding the `SimpleController_UsingActionAsset` script, there are some notable differences.

```CSharp
public class SimpleController_UsingActionAsset
{
    // This replaces the InputAction instances we had before with
    // the generated C# class.
    private SimpleControls m_Controls;

    //...

    public void Awake()
    {
        // To use the controls, we need to instantiate them.
        // This can be done arbitrary many times. E.g. there
        // can be multiple players each with its own SimpleControls
        // instance.
        m_Controls = new SimpleControls();

        // The generated C# class exposes all the action map
        // and actions in the asset by name. Here, we reference
        // the `fire` action in the `gameplay` action map, for
        // example.
        m_Controls.gameplay.fire.performed +=
        //...
    }

    //...

    public void Update()
    {
        // Same here, we can just look the actions up by name.
        var look = m_Controls.gameplay.look.ReadValue<Vector2>();
        var move = m_Controls.gameplay.move.ReadValue<Vector2>();

        //...
    }
}
```

Just for kicks, this sample also adds keyboard and mouse control to the game.

# SimpleDemo_UsingPlayerInput

[Source](./SimpleController_UsingPlayerInput.cs)

Finally, we reached the highest level of the input system. While scripting input like in the examples above can be quick and easy, it becomes hard to manage when there can be multiple devices and/or multiple players in the game. This is where `PlayerInput` comes in.

`PlayerInput` automatically manages per-player device assignments and can also automatically handle control scheme switching in single player (e.g. when the player switches between a gamepad and mouse&keyboard).

In our case, we're not getting too much out of it since we don't have control schemes or multiple players but still, let's have a look.

The first thing you'll probably notice is that now there are two script components on the `Player` object, one being the usual `SimpleController` and the other being `PlayerInput`. The latter is what now refers to [SimpleControls.inputactions](SimpleControls.inputactions). It also has `gameplay` set as the `Default Action Map` so that the gameplay actions will get enabled right away when `PlayerInput` itself is enabled.

For getting callbacks, we have chosen `Invoke Unity Events` as the `Behavior`. If you expand the `Events` foldout in the inspector, you can see that `OnFire`, `OnMove`, and `OnLook` are added to the respective events. Each callback method here looks like the `started`, `performed`, and `canceled` callbacks we've already seen on `fireAction` before.

```CSharp
public class SimpleController_UsingPlayerInput : MonoBehaviour
{
    private Vector2 m_Move;
    //...

    public void OnMove(InputAction.CallbackContext context)
    {
        m_Move = context.ReadValue<Vector2>();
    }

    //...
}
```
