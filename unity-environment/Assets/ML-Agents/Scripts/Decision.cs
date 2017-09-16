using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// Generic functions for Decision Interface
public interface Decision
{
    /// \brief Implement this method to define the logic of decision making 
    /// for the CoreBrainHeuristic
    /** Given the information about the agent, return a vector of actions.
	* @param state The state of the agent
	* @param observation The cameras the agent uses
	* @param reward The reward the agent had at the previous step
	* @param done Weather or not the agent is done
	* @param memory The memories stored from the previous step with MakeMemory()
	* @return The vector of actions the agent will take at the next step
	*/
    float[] Decide(List<float> state, List<Camera> observation, float reward, bool done, float[] memory);

    /// \brief Implement this method to define the logic of memory making for 
    /// the CoreBrainHeuristic
    /** Given the information about the agent, return the new memory vector for the agent.
	* @param state The state of the agent
	* @param observation The cameras the agent uses
	* @param reward The reward the agent had at the previous step
	* @param done Weather or not the agent is done
	* @param memory The memories stored from the previous step with MakeMemory()
	* @return The vector of memories the agent will use at the next step
	*/
    float[] MakeMemory(List<float> state, List<Camera> observation, float reward, bool done, float[] memory);
}