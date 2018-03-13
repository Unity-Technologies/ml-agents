# On Demand Decision Making

## Description
On demand decision making allows agents to request decisions from their 
brains only when needed instead of receiving decisions at a fixed 
frequency. This is useful when the agents commit to an action for a 
variable number of steps or when the agents cannot make decisions 
at the same time. This typically the case for turn based games, games 
where agents must react to events or games where agents can take 
actions of variable duration.

## How to use

To enable or disable on demand decision making, use the checkbox called
`On Demand Decisions` in the Agent Inspector.

<p align="center">
    <img src="images/ml-agents-ODD.png" 
        alt="On Demand Decision" 
        width="500" border="10" />
</p>

 * If `On Demand Decisions` is not checked, the Agent will request a new 
 decision every `Decision Frequency` steps and 
 perform an action every step. In the example above, 
 `CollectObservations()` will be called every 5 steps and 
 `AgentAction()` will be called at every step. This means that the 
 Agent will reuse the decision the Brain has given it. 

 * If `On Demand Decisions` is checked, the Agent controls when to receive
 decisions and actions. To do so, the Agent may leverage one or two methods:
   * `RequestDecision()` Signals that the Agent is requesting a decision.
   This causes the Agent to collect its observations and ask the Brain for a 
   decision at the next step of the simulation. Note that when an Agent 
   requests a decision, it also request an action. 
   This is to ensure that all decisions lead to an action during training.
   * `RequestAction()` Signals that the Agent is requesting an action. The
   action provided to the Agent in this case is the same action that was
   provided the last time it requested a decision. 
