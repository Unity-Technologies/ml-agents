# Heuristic Brain

The **Heuristic Brain** allows you to hand code an Agent's decision making
process. A Heuristic Brain requires an implementation of the Decision script
to which it delegates the decision making process.

When you use a **Heuristic Brain**, you must add a decision script to the `Decision` 
property of the **Heuristic Brain**.

## Implementing the Decision interface

```csharp
using UnityEngine;
using MLAgents;

public class HeuristicLogic : Decision
{
    // ...
}
```

The Decision interface defines two methods, `Decide()` and `MakeMemory()`.

The `Decide()` method receives an Agents current state, consisting of the
agent's observations, reward, memory and other aspects of the Agent's state, and
must return an array containing the action that the Agent should take. The
format of the returned action array depends on the **Vector Action Space Type**.
When using a **Continuous** action space, the action array is just a float array
with a length equal to the **Vector Action Space Size** setting. When using a
**Discrete** action space, the action array is an integer array with the same
size as the `Branches` array. In the discrete action space, the values of the
**Branches** array define the number of discrete values that your `Decide()`
function can return for each branch, which don't need to be consecutive
integers.

The `MakeMemory()` function allows you to pass data forward to the next
iteration of an Agent's decision making process. The array you return from
`MakeMemory()` is passed to the `Decide()` function in the next iteration. You
can use the memory to allow the Agent's decision process to take past actions
and observations into account when making the current decision. If your
heuristic logic does not require memory, just return an empty array.
