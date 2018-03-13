# On Demand Decision Making

## Description
On demand decision making allows agents to request decisions from their 
brains only when needed instead of requesting decisions at a fixed 
frequency. This is useful when the agents commit to an action for a 
variable number of steps or when the agents cannot make decisions 
at the same time. This typically the case for turn based games, games 
where agents must react to events or games where agents can take 
actions of variable duration.

## How to use

In the agent inspector, there is a checkbox called 
`On Demand Decision`

![Brain Inspector](images/ml-agents-ODD.png)

 * If `On Demand Decision` is not checked, all the agents will 
 request a new decision every `Decision Frequency` steps and 
 perform an action every step. In the example above, 
 `CollectObservations()` will be called every 5 steps and 
 `AgentAction()` will be called at every step. This means that the 
 agent will reuse the decision the brain has given it. 

 * If `On Demand Decision` is checked, you are in charge of telling 
 the agent when to request a decision and when to request an action. 
 To do so, call the following methods on your agent component.
   * `RequestDecision()` Call this method to signal the agent that it 
   must collect its observations and ask the brain for a decision at 
   the next step of the simulation. Note that when an agent requests 
   a decision, it will also request an action automatically 
   (This is to ensure that all decisions lead to an action during training)
   * `RequestAction()` Call this method to signal the agent that 
   it must reuse its previous action at the next step of the 
   simulation. The Agent will not ask the brain for a new decision, 
   it will just call `AgentAction()` with the same action.
