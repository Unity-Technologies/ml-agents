# How to Instantiate and Destroy Agents

In Unity, you can instantiate and destroy game objects but it can be tricky if this game object has an Agent component attached.

_Notice: This feature is still experimental._

## Instantiating an Agent
You will need another game object or agent to instantiate your agent. First you will need a prefab of the agent to instantiate it. You can use `Resource.Load()` or use a `public GameObject` field and drag your prefab into it. You must be careful to give a brain to the agent prefab. A lot of methods require the agent to have a brain, not having one can cause issues very rapidly. Fortunately, you can use the method `GiveBrain()` of the Agent to give it a brain. You should also call `AgentReset()` on this newly born agent so it will get the same start in life as his friends.

```csharp
agentPrefab.GetComponentInChildren<Agent>().GiveBrain(brain);
GameObject newAgent = Instantiate(agentPrefab);
agentPrefab.GetComponentInChildren<Agent>().RemoveBrain();
newAgent.GetComponentInChildren<Agent>().AgentReset();
```

Note that it is possible to generate an agent inside the `AgentStep()` method of an agent. Be careful, since the new agent could also create a new agent leading to an infinite loop.

## Destroying an Agent
Try not to destroy an agent by simply using `Destroy()`. This will confuse the learning process as the Brain will not know that the agent was terminated. The proper way to kill an agent is to set his done flag to `true` and make use of the `AgentOnDone()` method.  
In the default case, the agent resets when done but you can change this behavior. If you **uncheck** the `Reset On Done` checkbox of the agent, the agent will not reset and call instead `AgentOnDone()`. You must now implement the method `AgentOnDone()` as follows :

```csharp
public override void AgentOnDone()
{
    Destroy(gameObject);
}
```
This is the simplest case where you want the agent to be destroyed, but you can also do plenty of other things such as making and explosion, warn nearby agents, instantiate a zombie agent etc.
