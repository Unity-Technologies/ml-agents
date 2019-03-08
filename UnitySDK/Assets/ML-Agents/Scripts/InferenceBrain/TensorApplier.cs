using System.Collections.Generic;

namespace MLAgents.InferenceBrain
{    
    /// <summary>
    /// Mapping between the output Tensor names and the method that will use the
    /// output tensors and the Agents present in the batch to update their action, memories and
    /// value estimates.
    /// A TensorApplier implements a Dictionary of strings (node names) to an Action.
    /// This action takes as input the Tensor and the Dictionary of Agent to AgentInfo for
    /// the current batch.
    /// </summary>
    public class TensorApplier
    {
        /// <summary>
        /// A tensor Applier's Execute method takes a Tensor and a Dictionary of Agent to AgentInfo.
        /// Uses the data contained inside the Tensor to modify the state of the Agent. The Tensors
        /// are assumed to have the batch size on the first dimension and the agents to be ordered
        /// the same way in the dictionary and in the Tensor.
        /// </summary>
        public interface Applier
        {
            /// <summary>
            /// Applies the values in the Tensor to the Agents present in the agentInfos
            /// </summary>
            /// <param name="tensor"> The Tensor containing the data to be applied to the Agents</param>
            /// <param name="agentInfo"> Dictionary of Agents to AgentInfo that will reveive
            /// the values of the Tensor.</param>
            void Apply(Tensor tensor, Dictionary<Agent, AgentInfo> agentInfo);
        }
        
        Dictionary<string, Applier>  _dict = new Dictionary<string, Applier>();

        /// <summary>
        /// Returns a new TensorAppliers object.
        /// </summary>
        /// <param name="bp"> The BrainParameters used to determine what Appliers will be
        /// used</param>
        /// <param name="seed"> The seed the Appliers will be initialized with.</param>
        public TensorApplier(BrainParameters bp, int seed)
        {
            _dict[TensorNames.ValueEstimateOutput] = new ValueEstimateApplier();
            if (bp.vectorActionSpaceType == SpaceType.continuous)
            {
                _dict[TensorNames.ActionOutput] = new ContinuousActionOutputApplier();
            }
            else
            {
                _dict[TensorNames.ActionOutput] = new DiscreteActionOutputApplier(
                    bp.vectorActionSize, seed);
            }
            _dict[TensorNames.RecurrentOutput] = new MemoryOutputApplier();
            
            _dict[TensorNames.RecurrentOutput_C] = new BarracudaMemoryOutputApplier(true);
            _dict[TensorNames.RecurrentOutput_H] = new BarracudaMemoryOutputApplier(false);
        }

        /// <summary>
        /// Updates the state of the agents based on the data present in the tensor.
        /// </summary>
        /// <param name="tensors"> Enumerable of tensors containing the data.</param>
        /// <param name="agentInfos"> Dictionary of Agent to AgentInfo that contains the
        /// Agents that will be updated using the tensor's data</param>
        /// <exception cref="UnityAgentsException"> One of the tensor does not have an
        /// associated applier.</exception>
        public void ApplyTensors(
            IEnumerable<Tensor> tensors,  Dictionary<Agent, AgentInfo> agentInfos)
        {
            foreach (var tensor in tensors)
            {
                if (!_dict.ContainsKey(tensor.Name))
                {
                    throw new UnityAgentsException(
                        "Unknow tensor expected as output : "+tensor.Name);
                }
                _dict[tensor.Name].Apply(tensor, agentInfos);
            }
        }
    }
}
